# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import random
import tempfile
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from torch import Tensor, nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)

from nemo.collections.asr.models import EncDecSpeakerLabelModel

from nemo.collections.audio.parts.utils.resampling import resample
from nemo.core.classes.module import NeuralModule
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.models.duplex_s2s_model import replace_control_speech_codes, tokens_to_str
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.intelligibility import Intelligibility
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.metrics.secs import SECS
from nemo.collections.speechlm2.parts.metrics.token_accuracy import TokenAccuracy
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    set_model_dict_for_partial_init,
    setup_audio_codec,
    setup_speech_encoder,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging

from nemo.collections.tts.modules import transformer_2501

from nemo.collections.speechlm2.modules.cfm import MatchaTTSCFM
from types import SimpleNamespace



def get_mask_from_lengths(
    lengths: torch.Tensor = None,
    x: torch.Tensor = None,
    pad_to_factor: int = None
) -> torch.Tensor:
    """Constructs binary mask from a 1D torch tensor of input lengths
    Args:
        lengths: torch.tensor (torch.tensor): 1D tensor with lengths
        x: torch.tensor = tensor to be used on, last dimension is for mask
    Returns:
        mask (torch.tensor): num_sequences x max_length binary tensor
    """
    if lengths is None:
        assert x is not None
        return torch.ones(x.shape[-1], dtype=torch.bool, device=x.device)
    else:
        if x is None:
            max_len = torch.max(lengths)
        else:
            max_len = x.shape[-1]

    if pad_to_factor is not None:
        with fp32_precision():
            max_len = torch.ceil(max_len / pad_to_factor) * pad_to_factor

    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = ids < lengths.unsqueeze(1)
    return mask

def left_shift_input(tensor, pad_value=0.0):
    # Shifts left by 1, pads last timestep
    return torch.cat([tensor[:, 1:], torch.full_like(tensor[:, :1], pad_value)], dim=1)

def build_vocabs(subword_vocab: dict, subword_padding_idx: int, special_vocab: dict = None) -> tuple[dict, dict]:
    """
    Builds the character vocabulary and the mapping from subword ids to character ids.
    Args:
        subword_vocab (dict): A dictionary of subword vocab items. Eg.
            tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)
            subword_vocab = tokenizer.vocab
        subword_padding_idx (int): The padding index for the subword vocabulary.
        special_vocab (dict): items of special token dictionary (usually BOS, EOS)
            eg. special_vocab = {'<BOS>': 0, '<EOS>': 1}
    Returns:
        subword_id_to_char_ids: A dictionary mapping subword ids to character ids.
        char_vocab: A dictionary mapping character ids to their corresponding characters.
    """
    subword_vocab_items = subword_vocab.items() if isinstance(subword_vocab, dict) else subword_vocab
    org_char_vocab = {subword: subword_id for subword, subword_id in  subword_vocab_items if len(subword) == 1}

    # Add special tokens directly to char vocab
    if special_vocab is not None:
        for special_token, special_token_id in special_vocab.items():
            if special_token in org_char_vocab:
                raise ValueError(f"Special token {special_token} already exists in the character vocabulary.")
            org_char_vocab[special_token] = special_token_id

    sorted_char_vocab = dict(sorted(org_char_vocab.items(), key=lambda x: x[1]))
    char_vocab = {k: i for i, (k, _) in enumerate(sorted_char_vocab.items())}
    assert sorted(char_vocab.values()) == list(range(len(char_vocab)))
    subword_id_to_char_ids = {
        subword_id: tuple(char_vocab[char] for char in subword) for subword, subword_id in subword_vocab_items
    }

    # Creating mapping from subword ids of special tokens to their char ids
    if special_vocab is not None:
        for special_token, special_token_id in special_vocab.items():
            if special_token in subword_id_to_char_ids:
                raise ValueError(f"Special token {special_token} already exists in the subword id Vocabulary.")
            subword_id_to_char_ids[special_token_id] = (char_vocab[special_token],)

    assert max(subword_id_to_char_ids) == len(subword_id_to_char_ids) - 1

    # Always add padding token to the end of the vocab (this is the convention used in the original code)
    subword_id_to_char_ids[subword_padding_idx] = (len(char_vocab),)

    return subword_id_to_char_ids, char_vocab

def cosine_schedule(x: torch.Tensor):
    """
    Maps input values from [0, 1] to [1, 0] using the first quadrant of the cosine function.
    Used for MaskGit mask scheduling.
    """
    return torch.cos(x * (torch.pi / 2))

class CharAwareSubwordEncoder(NeuralModule):
    """
    Char-aware subword encoder for the MagpieTTS model.
    This module takes subword ids as input, maps them to character ids, and then applies a transformer encoder to the character embeddings.
    The output is a tensor of shape (batch_size, max_subword_length, d_embed).
    """
    def __init__(self, d_embed: int, llm_tokenizer_vocab: dict, subword_padding_idx: int, special_vocab: dict = None):
        """
        Args:
            d_embed (int): The dimension of the embedding.
            llm_tokenizer_vocab (dict): A dictionary of subword vocab items. Eg.
                tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)
                llm_tokenizer_vocab = tokenizer.vocab
            subword_padding_idx (int): The padding index for the subword vocabulary.
            special_vocab (dict): items of special token dictionary (usually BOS, EOS)
                eg. special_vocab = {'<BOS>': 30001, '<EOS>': 30002}
        """
        super().__init__()
        self.subword_id_to_char_ids, self.char_vocab = build_vocabs(llm_tokenizer_vocab, subword_padding_idx, special_vocab)
        self.embed_tokens = torch.nn.Embedding(self.vocab_size+1, d_embed, padding_idx=self.vocab_size)
        self.encoder = transformer_2501.Transformer(
            n_layers=1,
            d_model=d_embed,
            d_ffn=d_embed * 4,
            sa_n_heads=8,
            kernel_size=1,
            max_length_causal_mask=256,
            use_learnable_pos_emb=True
        )

    @property
    def vocab_size(self):
        return len(self.char_vocab)

    def prepare_inputs(self, subword_ids: Tensor, padding_mask: Tensor) -> tuple[Tensor, Tensor]:
        device = subword_ids.device

        subword_id_list = torch.masked_select(subword_ids, padding_mask).cpu().tolist()
        char_id_list = [list(self.subword_id_to_char_ids[x]) for x in subword_id_list]

        char_lengths = torch.tensor([len(x) for x in char_id_list], dtype=torch.long, device=device)
        batch_size = char_lengths.size(0)

        char_ids = torch.full((batch_size, int(char_lengths.max().item())), self.vocab_size, dtype=torch.long)
        for i in range(batch_size):
            char_ids[i, : char_lengths[i]] = torch.tensor(char_id_list[i])
        char_ids = char_ids.to(device=device)
        return char_ids, char_lengths

    def forward(self, subword_ids: Tensor, subword_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            subword_ids (Tensor): A tensor of shape (batch_size, max_subword_length) containing the subword ids.
            subword_mask (Tensor | None): A tensor of shape (batch_size, max_subword_length) containing the mask for the subword ids.
                If None, a mask of ones will be used.
        Returns:
            Tensor: A tensor of shape (batch_size, max_subword_length, d_embed) containing the subword embeddings.
        """
        device = subword_ids.device
        if subword_mask is None:
            subword_mask = torch.ones_like(subword_ids).bool()
        else:
            subword_mask = subword_mask.bool()

        if subword_mask.ndim == 3:
            subword_mask = subword_mask.squeeze(-1)

        char_ids, char_lengths = self.prepare_inputs(subword_ids, subword_mask)
        char_mask = get_mask_from_lengths(char_lengths)
        char_emb = self.embed_tokens(char_ids)
        # char emb has the shape  [B*T, N, channels], where N is the max number of chars tokens decoded from bpe tokens
        x = self.encoder(
            x=char_emb,
            x_mask=char_mask
        )['output']

        # Get average embedding over the chars
        mean_emb = ((x / char_mask.unsqueeze(-1).sum(1, keepdim=True)) * char_mask.unsqueeze(-1)).sum(1)
        subword_emb = torch.zeros((subword_mask.size(0), subword_mask.size(1), mean_emb.size(-1)), device=device)
        subword_emb[subword_mask.unsqueeze(-1).expand(-1, -1, mean_emb.size(-1))] = mean_emb.view(-1)
        
        return subword_emb

class GroupedCodec(NeuralModule):
    def __init__(self, codec, frame_stacking_factor):
        super().__init__()
        self.codec = codec
        self.frame_stacking_factor = frame_stacking_factor

    @property
    def device(self):
        return self.codec.device

    @property
    def _codebook_size(self):
        return self.codec.vector_quantizer.codebook_size_per_group

    @property
    def _num_codebooks(self):
        return self.codec.vector_quantizer.num_groups * self.frame_stacking_factor

    @property
    def samples_per_frame(self):
        return self.codec.samples_per_frame * self.frame_stacking_factor

    def encode(self, audio, audio_len):
        with fp32_precision():
            # make the audio divisible by frame rate and also by self.frame_stacking_factor with extra frames of 1 to avoid issues because we are removing a audio frame to shift target and input for TF
            audio, audio_len = self.pad_audio_to_factor(audio, audio_len, self.samples_per_frame, extra_frames=1)
            # encodes audio using the codec
            tokens, tokens_len = self.codec.encode(audio=audio, audio_len=audio_len)  # B, C, T
            tokens = tokens.transpose(1, 2)  # → B, T, C
            B, T, C = tokens.shape
            assert T % self.frame_stacking_factor == 0
            grouped = tokens.reshape(B, T // self.frame_stacking_factor, C * self.frame_stacking_factor)
            tokens_len = tokens_len // self.frame_stacking_factor
            return grouped.transpose(1, 2), tokens_len

    def decode(self, tokens, tokens_len):
        with fp32_precision():
            tokens = tokens.transpose(1, 2)
            # tokens: B, T', C'
            B, T, Cg = tokens.shape
            assert Cg % self.frame_stacking_factor == 0
            C = Cg // self.frame_stacking_factor
            ungrouped = tokens.reshape(B, T * self.frame_stacking_factor, C)  # → [B, T, C]
            ungrouped = ungrouped.transpose(1, 2)      # → [B, C, T] for decode
            tokens_len = torch.ceil(tokens_len * self.frame_stacking_factor).to(tokens_len.dtype)
            audio, audio_len = self.codec.decode(tokens=ungrouped, tokens_len=tokens_len)
        return audio, audio_len

    def decode_audio(self, inputs: torch.Tensor, input_len: torch.Tensor):
        """Apply decoder on the input. Note that the input is a non-quantized encoder output or a dequantized representation.

        Args:
            inputs: encoded signal
            input_len: valid length for each example in the batch

        Returns:
            Decoded output `audio` in the time domain and its length in number of samples `audio_len`.
            Note that `audio_len` will be a multiple of `self.samples_per_frame`.
        """
        with fp32_precision():
            if self.frame_stacking_factor > 1:
                inputs = inputs.transpose(1, 2)
                B, T, Cg = inputs.shape
                C = Cg // self.frame_stacking_factor
                inputs = inputs.reshape(B, T * self.frame_stacking_factor, C)  # → [B, T, C]
                input_len = torch.ceil(input_len * self.frame_stacking_factor).to(input_len.dtype)
                inputs = inputs.transpose(1, 2)

            audio, audio_len = self.codec.audio_decoder(inputs=inputs, input_len=input_len)
        return audio, audio_len

    def dequantize(self, tokens: torch.Tensor, tokens_len: torch.Tensor) -> torch.Tensor:
        """Convert the discrete tokens into a continuous encoded representation.

        Args:
            tokens: discrete tokens for each codebook for each time frame
            tokens_len: valid length of each example in the batch

        Returns:
            Continuous encoded representation of the discrete input representation.
        """
        with fp32_precision():
            # reshape to dequantize
            if self.frame_stacking_factor > 1:
                tokens = tokens.transpose(1, 2)
                # tokens: B, T', C'
                B, T, Cg = tokens.shape
                assert Cg % self.frame_stacking_factor == 0
                C = Cg // self.frame_stacking_factor
                tokens = tokens.reshape(B, T * self.frame_stacking_factor, C)  # → [B, T, C]
                tokens = tokens.transpose(1, 2)      # → [B, C, T] for decode
                tokens_len = torch.ceil(tokens_len * self.frame_stacking_factor).to(tokens_len.dtype)
            dequantized = self.codec.dequantize(tokens=tokens, tokens_len=tokens_len)
            # reshape back to the compress form if needed
            if self.frame_stacking_factor > 1:
                dequantized = dequantized.transpose(1, 2)  # → B, T, C
                B, T, C = dequantized.shape
                assert T % self.frame_stacking_factor == 0
                dequantized = dequantized.reshape(B, T // self.frame_stacking_factor, C * self.frame_stacking_factor)
                dequantized = dequantized.transpose(1, 2)  # → B, C, T

        return dequantized

    def forward(self, audio, audio_len):
        tokens, tokens_len = self.encode(audio, audio_len)
        audio, audio_len = self.decode(tokens, tokens_len)
        return audio, audio_len

    def pad_audio_to_factor(self, audio, audio_len, samples_per_frame, extra_frames: int = 0):
        """
        Zero pad the end of the audio so that we do not have a partial end frame.
        The output will be zero-padded to have an integer number of frames of
        length `samples_per_frame * frame_stacking_factor`.

        Args:
            audio: input time-domain signal (B, T)
            audio_len: valid length for each example in the batch (B,)
            samples_per_frame: number of samples per frame

        Returns:
            padded_audio: Padded time-domain signal (B, T')
            padded_len: Adjusted valid lengths (B,)
        """
        with fp32_precision():
            padded_len = (samples_per_frame * torch.ceil(audio_len / samples_per_frame).int()) + (extra_frames * samples_per_frame)
        max_len = padded_len.max().int().item()
        num_padding = (max_len - audio.shape[1])
        padded_audio = F.pad(audio, (0, num_padding))   
        return padded_audio, padded_len


class DecoderOnlyMagpieTTS(NeuralModule):
    def __init__(self, cfg: dict) -> None:
        # assert isinstance(cfg, dict), (
        #     "You must pass the config to DecoderOnlyMagpieTTS as a Python dict to support hyperparameter serialization "
        #     f"in PTL checkpoints (we got: '{type(cfg)=}')."
        # )
        super().__init__()
        # convert dict to config
        # cfg = DictConfig(cfg)
        self.config = cfg
        self.parallel_codebook_loss_scale = self.config.get("parallel_codebook_loss_scale", 1.0)
        self.config_unconditional_prob = self.config.get('cfg_unconditional_prob', 0.2)
        self.config_scale = self.config.get('cfg_scale', 2.5)
        self.use_local_transformer = self.config.get('use_local_transformer', True)
        self.local_transformer_type = self.config.get('local_transformer_type', "ar")
        self.local_transformer_loss_scale = self.config.get('local_transformer_loss_scale', 1.0)
        # ratio between the the codec frame rate and the Magpie decoder's frame rate
        self.downsampling_factor = self.config.get('downsampling_factor', 1)
        self.frame_stacking_factor = self.config.get('frame_stacking_factor', 1)
        self._num_codebooks = self.config.get('num_quantizers', 13)
        self._codebook_size = self.config.get('codebook_size', 2016)

        # Load ForCausalLM
        llm = load_pretrained_hf(self.config.pretrained_llm, pretrained_weights=self.config.get("pretrained_weights", True)).train()
        self.backbone = llm.model  # fetch PretrainedBaseModel from model "ForCausalLM"

        # Note: we have to "move out" the token embedding outside of LLM to avoid
        #       messing up FSDP/TP hooks.
        self.embed_text_tokens = self.backbone.embed_tokens
        del self.backbone.embed_tokens

        # audio embeddings
        audio_embeddings = []
        for _ in range(self._num_codebooks * self.downsampling_factor):
            audio_embeddings.append(nn.Embedding(self.speech_vocab_size, self.backbone.config.hidden_size))
        self.audio_embeddings = nn.ModuleList(audio_embeddings)

        # audio head
        self.final_proj = nn.Linear(self.backbone.config.hidden_size, self._num_codebooks * self.speech_vocab_size * self.downsampling_factor)

        # add loss
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

        # use BPE char aware tokenizer
        self.tokenizer = AutoTokenizer(self.config.pretrained_tokenizer, use_fast=True)
        llm_tokenizer_vocab_items = self.tokenizer.vocab
        # if vocab is a dict it already has the subword and token id, if not, get it from the tokenizer
        if isinstance(llm_tokenizer_vocab_items, dict):
            llm_tokenizer_vocab_items = llm_tokenizer_vocab_items.items()
        else:
            llm_tokenizer_vocab_items = [
                (subword, self.tokenizer.tokenizer._tokenizer.token_to_id(subword))
                for subword in llm_tokenizer_vocab_items
            ]

        self.cas_encoder = CharAwareSubwordEncoder(
            d_embed=self.backbone.config.hidden_size,
            llm_tokenizer_vocab=llm_tokenizer_vocab_items,
            subword_padding_idx=self.tokenizer.pad
        )

        # if use maskgit local transformer
        if self.use_local_transformer:
            if self.local_transformer_type == "cfm":
                self.codec_feature_dim = self._num_codebooks * 4
                # Instance new CFM based encoder
                decoder_params = {"channels": [256, 256], "dropout": 0.05, "attention_head_dim": 64, "n_blocks": 4, "num_mid_blocks": 2, "num_heads": 2, "act_fn": "snakebeta", "disable_down_up": True}
                cfm_params = SimpleNamespace(**{"solver": "euler", "sigma_min": 1e-4})

                # projection from model backbone to local transformer
                if self.codec_feature_dim != self.backbone.config.hidden_size:
                    self.local_transformer_in_projection = nn.Linear(self.backbone.config.hidden_size, self.codec_feature_dim)
                else:
                    self.local_transformer_in_projection = nn.Identity()

                self.local_transformer = MatchaTTSCFM(
                    in_channels=2 * self.codec_feature_dim,
                    out_channel=self.codec_feature_dim,
                    cfm_params=cfm_params,
                    decoder_params=decoder_params,
                    spk_emb_dim=0,
                )
            else:
                local_transformer_hidden_dim = self.config.get('local_transformer_hidden_dim', 768)
                self.local_transformer_mask_token_id = self.speech_vocab_size - 1# local transformer mask token

                # projection from model backbone to local transformer
                if local_transformer_hidden_dim != self.backbone.config.hidden_size:
                    self.local_transformer_in_projection = nn.Linear(self.backbone.config.hidden_size, local_transformer_hidden_dim)
                else:
                    self.local_transformer_in_projection = nn.Identity()

                self.local_transformer = transformer_2501.Transformer(
                    n_layers=self.config.get('local_transformer_n_layers', 4),
                    d_model=local_transformer_hidden_dim,
                    d_ffn=local_transformer_hidden_dim*4,
                    sa_n_heads=self.config.get('local_transformer_n_heads', 12),
                    kernel_size=1,
                    is_causal=True if self.local_transformer_type == "ar" else False,
                    max_length_causal_mask=self.downsampling_factor * self._num_codebooks+2,
                    use_learnable_pos_emb=True,
                )
                local_transformer_out_projections = []
                for _ in range(self._num_codebooks * self.downsampling_factor):
                    # Have a separate projection layer for each codebook, to distinguish between them
                    local_transformer_out_projections.append(nn.Linear(local_transformer_hidden_dim, self.speech_vocab_size))
                self.local_transformer_out_projections = nn.ModuleList(local_transformer_out_projections)

        # cached for quicker audio decoding
        self._use_fsdp = False
        self._use_tp = False

        # load pretrained TTS model
        if self.config.get("pretrained_model", None):
            self.init_model_from_another_checkpoint(self.config.pretrained_model)

    def init_model_from_another_checkpoint(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.state_dict())
            self.load_state_dict(checkpoint_state, strict=True)

    @property
    def speech_vocab_size(self):
        """Return the size of the audio codec codebook including extra speech BOS and EOS tokens."""
        if self.use_local_transformer and self.local_transformer_type == "nar": # add extra token for mask
            return self._codebook_size + 1
        return self._codebook_size + 1

    @property
    def speech_pad_id(self):
        return self._codebook_size

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def device(self):
        return next(self.parameters()).device

    def embed_audio_tokens(self, audio_tokens, lengths=None):
        audio_tokens = audio_tokens.transpose(1, 2).contiguous()
        B, C, T = audio_tokens.shape
        # Add and average the embeddings of the audio tokens across the codebooks
        audio_embedding = None
        for i in range(self.downsampling_factor):
            for c in range(C):
                tokens = audio_tokens[:,c , i::self.downsampling_factor]
                embedding = self.audio_embeddings[c + i * C](tokens)
                if audio_embedding is None:
                    audio_embedding = embedding
                else:
                    audio_embedding += embedding
        audio_embedding = audio_embedding / (C * self.downsampling_factor)

        # if lengths is not None return the new lenghts considering the downsampling_factor
        if lengths is not None:
            with fp32_precision():
                lengths = torch.ceil(lengths / self.downsampling_factor).to(lengths.dtype)
            return audio_embedding, lengths

        return audio_embedding

    def forward(
        self,
        code: Tensor,
        attention_mask: Tensor | None = None,
        subword_ids: Tensor | None = None,
        subword_mask: Tensor | None = None,
        audio_mask: Tensor | None = None,
        past_key_values = None,
        use_cache: bool = False,
        guidance_enabled: bool = True,
        teacher_forcing_inference: bool = False,
        **kwargs
    ) -> dict[str, Tensor]:
        """
        Separated text and speech prediction:
            - Speech prediction is achieved by a independent AR decoder based on last_hidden_state + audio tokens
            - For KV-cache:
                (1) llm cache depends on input cache is None or Not
                (2) speech_generation cache relys on reset_input_and_kv_cache function.
        """
        input_embeds = self.cas_encoder(subword_ids, subword_mask=subword_mask)  # (B, L, E)
        # shift input audio tokens by 1
        input_audio_tokens = F.pad(code[:, :-1], [0, 0, 1, 0], value=self.speech_pad_id)
        # get embedding
        input_audio_emb = self.embed_audio_tokens(
            input_audio_tokens
        )
        input_embeds = input_embeds + input_audio_emb

        if self.config_unconditional_prob and guidance_enabled and not teacher_forcing_inference:
            if self.training:
                # if training drop the "text" conditioning in a percentage of batch
                if torch.rand(1).item() < self.config_unconditional_prob:
                    # make the whole batch zeros to the unconditional model
                    input_embeds = torch.zeros_like(input_embeds)
            elif self.config_scale is not None and not self.training:
                # if inference or evaluation create a zero tensor for decoder input and concatenate it to compute unconditional logits
                input_embeds_zeros = torch.zeros_like(input_embeds)
                input_embeds = torch.cat([input_embeds, input_embeds_zeros], dim=0)
                # duplicate mask to match the new shape
                if attention_mask is not None:
                    attention_mask = torch.cat([attention_mask, attention_mask], dim=0)

        out = self.backbone(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=use_cache, return_dict=True
        )
        B, T = input_embeds.shape[:2]

        # get logits
        logits = self.final_proj(out.last_hidden_state)  # (B, T', num_codebooks * _codebook_size)
        
        # inference/model initialization
        if audio_mask is not None and not self.training and not teacher_forcing_inference:
            return {"past_key_values": out.past_key_values}

        # teacher forcing inference
        if audio_mask is not None and not self.training and teacher_forcing_inference:
            # local transformer
            local_transformer_logits = None
            if self.use_local_transformer and self.local_transformer_type != "cfm":
                if self.local_transformer_type == "ar":
                    # autoregressive
                    local_transformer_logits = self.compute_local_transformer_logits(out.last_hidden_state, code, targets_offset_by_one=False)
                else:
                    # randomly replace some positions with MASK_TOKEN
                    audio_codes_masked, mask_tokens_mask = self.maskgit_apply_random_mask(code)
                    local_transformer_logits = self.compute_local_transformer_logits(out.last_hidden_state, code, targets_offset_by_one=True)

                lengths = torch.tensor([code.shape[1]] * code.shape[0], device=self.device)
                codes = self.logits_to_audio_codes(local_transformer_logits, lengths).transpose(1, 2)
            else:
                lengths = torch.tensor([code.shape[1]] * code.shape[0], device=self.device)
                codes = self.logits_to_audio_codes(out.last_hidden_state, lengths).transpose(1, 2)
            return {"codes": codes}
        
        # training time compute losses
        if audio_mask is not None:
            codebook_loss, loss_mask = self.compute_loss(logits, code, loss_mask=audio_mask)
            # local transformer
            local_transformer_logits = None
            if self.use_local_transformer:
                if self.local_transformer_type == "ar":
                    # autoregressive
                    local_transformer_logits = self.compute_local_transformer_logits(out.last_hidden_state, code, targets_offset_by_one=False)
                    local_transformer_loss, _ = self.compute_loss(local_transformer_logits, code, loss_mask=audio_mask)
                elif self.local_transformer_type == "cfm":
                    # move time dimention to batch
                    encoded = out.last_hidden_state.reshape(-1, out.last_hidden_state.size(-1)).unsqueeze(1)
                    # remove special tokens from labels
                    audio_labels = replace_control_speech_codes(code, self._control_codes).transpose(1, 2)
                    # get codec latent from audio tokens
                    codec_latent = self.audio_codec.dequantize(audio_labels, inputs["output_lens"]).detach().transpose(1, 2)
                    # move time dimention to batch
                    codec_latent = codec_latent.to(encoded.dtype).reshape(-1, codec_latent.size(-1)).unsqueeze(-1)
                    mask = inputs["audio_mask"].reshape(codec_latent.size(0)).unsqueeze(1).unsqueeze(-1)
                    encoded = self.local_transformer_in_projection(encoded).transpose(1, 2)
                    local_transformer_loss, _  = self.local_transformer.compute_loss(x1=codec_latent, mask=mask, mu=encoded, spks=None)
                else:
                    # randomly replace some positions with MASK_TOKEN
                    audio_codes_masked, mask_tokens_mask = self.maskgit_apply_random_mask(code)
                    local_transformer_logits = self.compute_local_transformer_logits(out.last_hidden_state, code, targets_offset_by_one=True)
                    local_transformer_loss, _ =  self.compute_loss(local_transformer_logits, code, mask_tokens_mask=mask_tokens_mask, loss_mask=audio_mask)
            else:
                local_transformer_loss = torch.tensor(0.0, device=self.device)

            loss_dict = {
                "local_transformer_loss": local_transformer_loss *  self.local_transformer_loss_scale,
                "codebook_loss": codebook_loss * self.parallel_codebook_loss_scale,
            }
            return {"loss_dict": loss_dict, "backbone_out": out.last_hidden_state, "past_key_values": out.past_key_values}

        # inference time
        # if using cfg and it is in inference or evaluation mix unconditional and coditional logits
        if self.config_scale is not None and self.config_unconditional_prob and not self.training and guidance_enabled:
            batch_size = logits.size(0) // 2
            cond_logits = logits[:batch_size]
            uncond_logits = logits[batch_size:]
            logits = (1 - self.config_scale) * uncond_logits + self.config_scale * cond_logits

        if self.use_local_transformer and self.local_transformer_type != "cfm":
            codes = self.local_transformer_sample_codes_from_logits(out.last_hidden_state[:, -1], temperature=self.config.get('temperature', 0.7), topk=self.config.get('topk', 80)) # (B, num_codebooks)
        else:
            codes = self.sample_codes_from_logits(logits[:, -1], temperature=self.config.get('temperature', 0.7), topk=self.config.get('topk', 80)) # (B, num_codebooks)


        ans = {
            "codes": codes.transpose(1, 2),
            "past_key_values": out.past_key_values,
        }
        return ans

    def pad_audio_codes_to_factor(self, audio_codes: torch.Tensor, downsampling_factor: int = 1, pad_token: int = 0):
        """
        Pads the time dimension of the audio codes to a multiple of the downsampling factor.
        Args:
            audio_codes (torch.Tensor): B, C, T
            downsampling_factor (int): The factor to downsample by.
            pad_token (int): The token ID to pad with.
        Returns:
            B, C, T_padded
        """
        audio_codes = audio_codes.transpose(1, 2)
        T = audio_codes.size(2)
        with fp32_precision():
            T_padded = (torch.ceil(torch.tensor(T / downsampling_factor)) * downsampling_factor).int().item()

        if T_padded > T:
            padding = pad_token * torch.ones(audio_codes.size(0), audio_codes.size(1), T_padded - T, device=audio_codes.device, dtype=audio_codes.dtype)
            audio_codes = torch.cat([audio_codes, padding], dim=2)
        return audio_codes.transpose(1, 2)


    def compute_loss(self, logits, audio_codes, audio_codes_lens=None, mask_tokens_mask=None, loss_mask=None):
        """
        Computes the audio codebook loss. Used by
        (1) The main Magpie-TTS transformer
        (2) The local transformer, for both autoregressive and MaskGit methods

        logits: (B, T', num_codebooks * num_tokens_per_codebook)
        audio_codes: (B, C, T')
        audio_codes_lens: (B,)
        mask_tokens_mask: (B, C, T') True for tokens that were replaced with the MASK_TOKEN and should
                                     therefore be the only ones included in the loss computation (for MaskGit).
        """
        with loss_parallel():
            audio_codes = audio_codes.transpose(1, 2).long()
            if loss_mask is None:
                loss_mask = get_mask_from_lengths(audio_codes_lens, pad_to_factor=self.downsampling_factor)

            if mask_tokens_mask is not None:
                loss_mask = loss_mask.unsqueeze(1) * mask_tokens_mask
                if not loss_mask.any():
                    # Without this we were very rarely getting NaNs in the loss
                    logging.warning("No tokens valid were found in compute_loss()!")
                    return torch.tensor(0.0, device=loss_mask.device), loss_mask
            else:
                # repeat loss mask for each codebook to simplify code below
                loss_mask = loss_mask.unsqueeze(1).repeat(1, audio_codes.size(1), 1)

            total_codebook_loss = None
            for ds_index in range(self.downsampling_factor):
                for codebook in range(audio_codes.size(1)):
                    si = (codebook + self._num_codebooks * ds_index) * self.speech_vocab_size
                    ei = si + self.speech_vocab_size
                    codebook_logits = logits[:, :, si:ei]  # (B, T', num_tokens_per_codebook)
                    codebook_targets = audio_codes[:, codebook, ds_index::self.downsampling_factor]  # (B, T')
                    codebook_loss = self.cross_entropy_loss(
                        codebook_logits.permute(0, 2, 1), codebook_targets  # (B, num_tokens_per_codebook, T')
                    )  # (B, T')
                    codebook_loss_mask = loss_mask[:, codebook, ds_index::self.downsampling_factor]
                    codebook_loss = codebook_loss * codebook_loss_mask
                    if codebook_loss_mask.sum() == 0:
                        logging.warning(f"Loss mask for codebook {codebook} is all zeros, global_step: {self.global_step}")
                        continue
                    codebook_loss = codebook_loss.sum() / codebook_loss_mask.sum()
                    if total_codebook_loss is None:
                        total_codebook_loss = codebook_loss
                    else:
                        total_codebook_loss = total_codebook_loss + codebook_loss

            total_codebook_loss = total_codebook_loss / (audio_codes.size(1) * self.downsampling_factor) 
        return total_codebook_loss, loss_mask


    def compute_local_transformer_logits(self, dec_out, audio_codes_target, targets_offset_by_one=False):
        """
        Predicts the logits for all codebooks using the local transformer. Used in both autoregressive (AR) and MaskGit (MG) modes.
        This function is used in training and validation, not inference/sampling.
        The sequence layout is slightly different between AR and MG modes, as shown in the diagram below,
        (using an 8-codebook setup as an example):
        +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
        | AR target  |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |   none  |
        | codebook   |         |         |         |         |         |         |         |         |         |
        +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
        | MG target  |  none   |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |
        | codebook   |         |         |         |         |         |         |         |         |         |
        +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
        |  input     | Magpie  |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |
        |  codebook  | latent  | or MASK | or MASK | or MASK | or MASK | or MASK | or MASK | or MASK | or MASK |
        +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
        | seq. index |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |    8    |
        +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
        
        dec_out: (B, T'/downsampling_factor, E)
        audio_codes_target: (B, C, T')
        targets_offset_by_one: bool, if False, the target for index 0 is codebook 0, for index 1 is codebook 1, etc. (autoregressive)
                                     if True,  the target for index 1 is codebook 0, for index 2 is codebook 1, etc. 
                                     (MaskGit)
        """
        audio_codes_target = audio_codes_target.transpose(1, 2)
        C = audio_codes_target.size(1)
        dec_out_all = dec_out.reshape(-1, dec_out.size(-1)) # (B*T', E)
        local_transformer_input = [dec_out_all]
        for ds_index in range(self.downsampling_factor):
            for codebook_num in range(C):
                codes = audio_codes_target[:, codebook_num, ds_index::self.downsampling_factor] # (B, T')
                codes = codes.reshape(-1) # (B*T',)
                codebook_embedding = self.audio_embeddings[codebook_num + ds_index * C](codes) # (B*T', E)
                local_transformer_input.append(codebook_embedding)

        local_transformer_input = torch.stack(local_transformer_input, dim=1) # (B*T', C+1, E)
        local_transformer_input = self.local_transformer_in_projection(local_transformer_input) # (B*T', C+1, 128)
        _mask = torch.ones(local_transformer_input.size(0), local_transformer_input.size(1), device=local_transformer_input.device)
        local_transformer_output = self.local_transformer(local_transformer_input, _mask)['output'] # (B*T', C+1, E)
        if not targets_offset_by_one:
            # for autoregressive local transformer the target for index 0 is codebook 0, for index 1 is codebook 1, etc.
            local_transformer_output = local_transformer_output[:, :-1, :] # (B*T', C, E)
        else:
            # for MaskGit the target for index **1** is codebook 0, for index 2 is codebook 1, etc.
            local_transformer_output = local_transformer_output[:, 1:, :] # (B*T', C, E)
        all_code_logits = []
        for ds_index in range(self.downsampling_factor):
            for codebook_num in range(audio_codes_target.size(1)):
                # Using a separate projection layer for each codebook (to distinguish between them)
                # Checked the time - this loop is not taking much time (compared to the local transformer forward pass)
                codebook_logits = self.local_transformer_out_projections[codebook_num + ds_index*C](local_transformer_output[:, codebook_num + ds_index*C, :]) # (B*T', num_all_tokens_per_codebook)
                all_code_logits.append(codebook_logits)

        # TODO @rfejgin: make sure all downsampling_factor * num_codebooks are in the same dimension (expected by loss calculation)
        all_code_logits = torch.cat(all_code_logits, dim=1) # (B*T'/downsampling_factor, num_codebooks * num_all_tokens_per_codebook * downsampling_factor)
        all_code_logits = all_code_logits.view(
            audio_codes_target.size(0), audio_codes_target.size(2) // self.downsampling_factor, -1
        ) # (B, T'/downsampling_factor, C * num_all_tokens_per_codebook * downsampling_factor)

        return all_code_logits


    def maskgit_create_random_mask(self, codes):
        """
        Creates a mask where True indicates the positions that should be replaced with a MASK_TOKEN.
        """
        # Codes: (B, C, T)
        B,C,T = codes.shape
        # get a uniform random vector uniformly sampled from [0,1) ## Todo does it need to be inclusive on the right?
        rand_values = torch.rand(B,T, device=codes.device)
        # apply the cosine schedule
        frac_masked = cosine_schedule(rand_values)
        # how many positions to mask
        n_masked = torch.ceil(frac_masked * C).long() # B,T
        # start from all unmasked
        mask = torch.zeros_like(codes, dtype=torch.bool)
        # The code further below is the vectorized version of this:
        #  for b in range(B):
        #      for t in range(T):
        #          if n_masked[b,t] > 0:
        #              # get a random permutation of the codebook indices
        #              perm = torch.randperm(C)
        #              # mask the top n_masked positions
        #              mask[b, perm[:n_masked[b,t]], t] = True
        #
        # Create random permutations 
        random_permutations = torch.argsort(torch.rand(B, C, T, device=codes.device), dim=1)  # (B, C, T)        
        # Create a mask tensor where each position indicates if it should be masked        
        mask_indices = torch.arange(C, device=codes.device).view(1, C, 1)
        mask = mask_indices < n_masked.view(B, 1, T) # (B, C, T)
        # Apply the random permutations to the mask
        mask = torch.gather(mask, 1, random_permutations)

        return mask # (B, C, T).

    def maskgit_apply_random_mask(self, codes):
        # Randomly replaces some codes with the MASK_TOKEN with a proportion following the cosine schedule.
        # Codes: (B, C, T)
        codes = codes.transpose(1, 2)
        mask = self.maskgit_create_random_mask(codes)
        ## replace some tokens with MASK_TOKEN
        codes_with_mask = torch.where(mask, self.local_transformer_mask_token_id, codes)
        return codes_with_mask, mask

    def logits_to_audio_codes(self, all_code_logits, audio_codes_lens):
        # all_code_logits: (B, T', num_codebooks * num_tokens_per_codebook)
        # audio_codes_lens: (B,)
        all_preds = [[] for _ in range(self.downsampling_factor)]
        for ds_index in range(self.downsampling_factor):
            for idx in range(self._num_codebooks):
                si = (idx + self._num_codebooks * ds_index) * self.speech_vocab_size
                ei = si + self.speech_vocab_size
                codebook_logits = all_code_logits[:, :, si:ei]
                codebook_probs = torch.softmax(codebook_logits, dim=-1)  # (B, T', num_tokens_per_codebook)
                # argmax to get the tokens
                codebook_preds = torch.argmax(codebook_probs, dim=-1)  # (B, T')
                # codebook_preds = target_audio_codes[:, idx, ds_index::self.downsampling_factor]
                all_preds[ds_index].append(codebook_preds)
        all_preds = [torch.stack(p, dim=1) for p in all_preds] # list of `downsampling_factor`` elements of shape (B,C,T) each
        all_preds = torch.stack(all_preds, dim=-1) # B, C, T, downsampling_factor
        # interleave the time dimension, undoing the frame stacking
        all_preds = all_preds.reshape(all_preds.size(0), all_preds.size(1), -1) # B, C, T*downsampling_factor
        pred_max_len = all_preds.size(2)
        real_max_len = audio_codes_lens.max()
        assert (pred_max_len - real_max_len) < self.downsampling_factor
        # trim padding introduced for frame stacking
        all_preds = all_preds[:,:, :real_max_len]
        audio_mask = get_mask_from_lengths(audio_codes_lens)
        all_preds = all_preds * audio_mask.unsqueeze(1)

        return all_preds

    def sample_codes_from_logits(self, all_code_logits_t, temperature=0.7, topk=80, unfinished_items={}, finished_items={}):
        # all_code_logits_t: (B, num_codebooks * num_tokens_per_codebook), logits at a given timestep
        all_preds = [[] for _ in range(self.downsampling_factor)]
        for ds_index in range(self.downsampling_factor):
            for idx in range(self._num_codebooks):
                si = (idx + self._num_codebooks * ds_index) * self.speech_vocab_size
                ei = si + self.speech_vocab_size
                codebook_logits = all_code_logits_t[:, si:ei]  # (B, num_tokens_per_codebook)
                # for item_idx in unfinished_items:
                #     codebook_logits[item_idx, self.speech_eos_id] = float('-inf')
                # for item_idx in finished_items:
                #     codebook_logits[item_idx, :] = float('-inf')
                #     codebook_logits[item_idx, self.speech_eos_id] = 0.0
                codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0]  # (B, topk)
                indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(
                    -1
                )  # (B, num_tokens_per_codebook)
                codebook_logits_rescored = codebook_logits.clone()
                codebook_logits_rescored[indices_to_remove] = float('-inf')

                codebook_probs = torch.softmax(codebook_logits_rescored / temperature, dim=-1)  # (B, num_tokens_per_codebook)
                codebook_preds = torch.multinomial(codebook_probs, 1)  # (B, 1)
                all_preds[ds_index].append(codebook_preds)
                
        all_preds = [torch.cat(ds_preds, dim=1).long() for ds_preds in all_preds]  # list of downsampling_factor of (B, num_codebooks)

        all_preds = torch.stack(all_preds, dim=2)  # (B, num_codebooks, downsampling_factor)

        # cat to get the right shape
        # all_preds = torch.cat(all_preds, dim=-1)# (B, num_codebooks)
        return all_preds

    def local_transformer_sample_maskgit(self, dec_output, temperature=0.7, topk=80, unfinished_items={}, finished_items={}, use_cfg=False, cfg_scale=1.0, n_steps=3, noise_scale=0.0, fixed_schedule_n_unmasked=None, dynamic_cfg_scale=False, sampling_type=None):
        """
        Sample codes for one timestep from the local transformer using MaskGit.
        """
        # ToDo: Fix maskgit inference error
  
        debug_print = False
        # dec_output: (B, E)
        device = dec_output.device
        # disable KV cache since our transformer is not causal
        self.local_transformer.reset_cache(use_cache=False)
        dec_output = dec_output.unsqueeze(1) # (B, 1, E)
        local_transformer_input_init = self.local_transformer_in_projection(dec_output) # (B, 1, D) where D is the dimension of the local transformer
        codebook_seq_len = self._num_codebooks * self.downsampling_factor
        B = dec_output.size(0)

        min_confidence = 0
        # this needs to be large enough that unmasked items will always remain unmasked (even after noise addition)
        # Setting it smaller could allow "regret", i.e. re-masking a codebook that was previously unmasked; we might want to try that
        max_confidence = 5 
        confidences = min_confidence * torch.ones(B, codebook_seq_len, device=device)
        # initialize to all masked
        codes = self.local_transformer_mask_token_id * torch.ones((B, codebook_seq_len), device=device, dtype=torch.long)
        sampled_codes = codes.clone()
        topk_indices = None
        if debug_print: print(f"Sampling type: {sampling_type}")
        if fixed_schedule_n_unmasked is not None:
            n_steps = len(fixed_schedule_n_unmasked)
            if debug_print: print(f"Using fixed schedule: {fixed_schedule_n_unmasked}")        
        if dynamic_cfg_scale:
            if debug_print: print(f"Using dynamic CFG scale")
        for step in range(n_steps):
            # how far along we are in the unmasking process
            progress = step / n_steps
            # get mask fraction
            frac_masked = cosine_schedule(torch.tensor(progress))
            if sampling_type == "causal":
                frac_masked = torch.ones_like(frac_masked) * (1.0 - progress)
            # how many codebooks to mask
            if fixed_schedule_n_unmasked is None:
                n_masked = torch.ceil(codebook_seq_len * frac_masked).long()
            else:
                n_masked = codebook_seq_len - fixed_schedule_n_unmasked[step]
            n_unmasked = codebook_seq_len - n_masked

            if sampling_type == "causal":# and n_unmasked <= self._num_codebooks:
                # force second frame not to be unmasked
                n_frames_to_allow = int(np.floor(progress*self.downsampling_factor+1))
                confidences[:,n_frames_to_allow*self._num_codebooks:] = min_confidence-1 # only works for downsampling_factor=2
            elif sampling_type == "alternate":
                # TODO: preserve already-unmasked codebooks
                if step % 2 == 1:
                    confidences[:,self._num_codebooks:] = min_confidence-1
                else:
                    confidences[:,:self._num_codebooks] = min_confidence-1
                # for unmasked codebooks, set confidence to max so that they will remain unmasked
                if topk_indices is not None:
                    confidences.scatter_(index=topk_indices, dim=1, src=max_confidence*torch.ones_like(topk_indices, dtype=confidences.dtype))

            # pick top-confidence codebooks up to n_unmasked
            _, topk_indices = torch.topk(confidences, k=n_unmasked, dim=1)
            if use_cfg:
                actual_batch_size = topk_indices.size(0) // 2
                assert (topk_indices[actual_batch_size:] == topk_indices[:actual_batch_size]).all(), f"Topk indices are not the same for conditional and unconditional codes"

            # replace masks of the top-k confident codebooks with the codes that were sampled for them
            unmasked_codes = torch.gather(sampled_codes, dim=1, index=topk_indices)
            codes.scatter_(dim=1, index=topk_indices, src=unmasked_codes)
            if debug_print:
                print(f"Transformer Input at step {step} of {n_steps}")
                self.vis_codes(codes, mask_id=self.local_transformer_mask_token_id, downsample_rate=self.downsampling_factor)
                print("--------------------------------")

            # build transformer input
            local_transformer_input = local_transformer_input_init
            for codebook_num in range(codebook_seq_len):
                next_local_transformer_input = self.audio_embeddings[codebook_num](codes[:, codebook_num]).unsqueeze(1) # (B, 1, 768)
                next_local_transformer_input = self.local_transformer_in_projection(next_local_transformer_input) # (B, 1, d_local)
                local_transformer_input = torch.cat([local_transformer_input, next_local_transformer_input], dim=1) # (B, codebook_num+1, d_local)

            # run transformer
            _mask = torch.ones(B, codebook_seq_len+1, device=device)
            local_transformer_output = self.local_transformer(local_transformer_input, _mask)['output'] # (B, C+1, d_local)

            # get logits
            logits = []
            for codebook_num in range(codebook_seq_len):
                # The `codebook_num+1` is to drop first position which corresponds to the magpie latent
                codebook_logits = self.local_transformer_out_projections[codebook_num](local_transformer_output[:, codebook_num+1, :]) # (B, num_audio_tokens_per_codebook)
                logits.append(codebook_logits)
            logits = torch.stack(logits, dim=1) # (B, C*downsampling_factor, num_audio_tokens_per_codebook)

            # apply CFG
            if use_cfg:
                actual_batch_size = logits.size(0) // 2
                conditional_logits = logits[:actual_batch_size]
                unconditional_logits = logits[actual_batch_size:]
                if not dynamic_cfg_scale:
                    current_cfg_scale = cfg_scale
                else:
                    # gradually increase the scale until mid point through sampling, then reduce it again
                    progress = step / (n_steps-1)
                    #interp = -abs(progress-0.5)+0.5 # increase from 0..1 in the interval from start to midpoint and then go back to zero
                    #interp = 1.0 - progress  # decrease from 1 to 0
                    interp = progress # gradually increase from 0 to 1
                    current_cfg_scale = (cfg_scale - 1) * interp + 1.0  # 1.0 --> cfg_scale --> 1.0
                    print(f"step={step}, current_cfg_scale={current_cfg_scale:.2f}")
                cfg_logits = current_cfg_scale * conditional_logits +  (1.0 - current_cfg_scale) * unconditional_logits                                    
                logits[:actual_batch_size] = cfg_logits

            # handle unfinished and finished items
            for item_idx in unfinished_items:
                logits[item_idx, self.audio_eos_id] = float('-inf')
            for item_idx in finished_items:
                logits[item_idx, :, :] = float('-inf')
                logits[item_idx, :, self.audio_eos_id] = 0.0

            # HACK: disallow generation of MASK tokens. But is this happening?
            logits[:,:,self.local_transformer_mask_token_id] = -200
            # logits[:,:,self.speech_bos_id] = -200

            # sample with top-k
            logits_topk = torch.topk(logits, topk, dim=-1)[0] # (B, C, topk)
            indices_to_remove = logits < logits_topk[:, :, -1].unsqueeze(-1) # (B, C, num_audio_tokens_per_codebook)
            logits_rescored = logits.clone()
            logits_rescored[indices_to_remove] = float('-inf')
            probs = torch.softmax(logits_rescored / temperature, dim=-1) # (B, C, num_audio_tokens_per_codebook)
            sampled_codes = torch.multinomial(probs.view(B*codebook_seq_len, -1), 1).view(B, codebook_seq_len)
            if use_cfg:
                sampled_codes[actual_batch_size:] = sampled_codes[:actual_batch_size]
                probs[actual_batch_size:] = probs[:actual_batch_size]
            confidences = torch.gather(probs, dim=2, index=sampled_codes.unsqueeze(-1)).squeeze(-1)

            # TODO
            # * are end of utterance-logits-somehow overwritten ? should we force those to max confidence?? may require
            #   special handling and may explain termination issues!

            # replace entries in sampled_codes with previously unmasked codebooks
            sampled_codes.scatter_(dim=1, index=topk_indices, src=unmasked_codes)
            #  add noise to confidences (as in token-critic paper, https://arxiv.org/abs/2209.04439)
            if noise_scale > 0.0:
                # get noise from uniform distribution in the interval [-0.5, 0.5), scale it by `noise_scale`,
                # and anneal it to 0 as we approach the end of the unmasking process
                noise = (torch.rand_like(confidences) - 0.5) * noise_scale * (1-(step+2)/n_steps) # the +2 makes sure that by the last iteration the noise is exactly 0
                confidences += noise
                # the conditional and unconditional get different noise and must be fixed to be the same again
                confidences[actual_batch_size:] = confidences[:actual_batch_size]                
            confidence_eps = 0.1
            assert confidences.max() + confidence_eps < max_confidence, f"Predicted confidence is approaching max_confidence: {confidences.max()}"
            # for unmasked codebooks, set confidence to max so that they will remain unmasked
            confidences.scatter_(index=topk_indices, dim=1, src=max_confidence*torch.ones_like(topk_indices, dtype=confidences.dtype))
        rand_sampling = False
        if rand_sampling:
            print("Using random sampling")
            confidences = torch.rand_like(confidences)
        codes = sampled_codes
        assert not (codes == self.local_transformer_mask_token_id).any(), f"Codes contain mask tokens after completion of MaskGit sampling"

        if debug_print:
            print(f"Final codes after MaskGit sampling")
            self.vis_codes(codes, mask_id=self.local_transformer_mask_token_id, downsample_rate=self.downsampling_factor)
            print("--------------------------------")
        # break downsampled groups of frames into individual frames
        codes = codes.reshape(B, self.downsampling_factor, self._num_codebooks).permute(0,2,1) # B, C, downsampling_factor

        if use_cfg: 
            # drop unconditional codes
            codes = codes[:actual_batch_size]
        return codes

    def local_transformer_sample_autoregressive(self, dec_output, temperature=0.7, topk=80, unfinished_items={}, finished_items={}, use_cfg=False, cfg_scale=1.0):
        # dec_output: (B, E)
        self.local_transformer.reset_cache(use_cache=True)
        dec_output = dec_output.unsqueeze(1) # (B, 1, E)
        local_transformer_input = self.local_transformer_in_projection(dec_output) # (B, 1, 128)
        all_preds = []
        for codebook_num in range(self._num_codebooks * self.downsampling_factor):
            _mask = torch.ones(local_transformer_input.size(0), local_transformer_input.size(1), device=local_transformer_input.device)
            local_transformer_output = self.local_transformer(local_transformer_input, _mask)['output'] # (B, T, 128)
            codebook_logits = self.local_transformer_out_projections[codebook_num](local_transformer_output[:, -1, :]) # (B, num_all_tokens_per_codebook)
            if use_cfg:
                actual_batch_size = codebook_logits.size(0) // 2
                conditional_logits = codebook_logits[:actual_batch_size]
                unconditional_logits = codebook_logits[actual_batch_size:]
                cfg_logits = cfg_scale * conditional_logits +  (1.0 - cfg_scale) * unconditional_logits
                codebook_logits[:actual_batch_size] = cfg_logits

            for item_idx in unfinished_items:
                codebook_logits[item_idx, self.audio_eos_id] = float('-inf')
            for item_idx in finished_items:
                codebook_logits[item_idx, :] = float('-inf')
                codebook_logits[item_idx, self.audio_eos_id] = 0.0

            codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0] # (B, topk)
            indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(-1) # (B, num_tokens_per_codebook)
            codebook_logits_rescored = codebook_logits.clone()
            codebook_logits_rescored[indices_to_remove] = float('-inf')
            codebook_probs = torch.softmax(codebook_logits_rescored / temperature, dim=-1) # (B, num_tokens_per_codebook)
            codebook_preds = torch.multinomial(codebook_probs, 1) # (B, 1)
            if use_cfg:
                codebook_preds[actual_batch_size:] = codebook_preds[:actual_batch_size]
            all_preds.append(codebook_preds)
            next_local_transformer_input = self.audio_embeddings[codebook_num](codebook_preds.squeeze(-1)).unsqueeze(1) # (B, 1, 128)
            next_local_transformer_input = self.local_transformer_in_projection(next_local_transformer_input) # (B, 1, 128)
            local_transformer_input = torch.cat([local_transformer_input, next_local_transformer_input], dim=1) # (B, T+1, 128)

        all_preds = torch.cat(all_preds, dim=1).long() # (B, num_codebooks * downsampling_factor)
        all_preds = all_preds.reshape(-1, self.downsampling_factor, self._num_codebooks).permute(0,2,1) # (B, num_codebooks, downsampling_factor)
        if use_cfg:
            all_preds = all_preds[:actual_batch_size]

        self.local_transformer.reset_cache(use_cache=False)
        return all_preds

    def local_transformer_sample_codes_from_logits(self, backbone_out, temperature=0.7, topk=80, unfinished_items={}, finished_items={}, dynamic_cfg_scale=False, maskgit_n_steps=4, maskgit_noise_scale=0.0, maskgit_sampling_type="causal", fixed_schedule_n_unmasked=None):

        if self.local_transformer_type == "ar" :
            # Autoregressive sampling with local transformer
            gen_tokens = self.local_transformer_sample_autoregressive(
                dec_output=backbone_out,
                temperature=temperature,
                topk=topk,
                use_cfg=self.config_unconditional_prob,
                cfg_scale=self.config_scale
            )

        else:
            gen_tokens = self.local_transformer_sample_maskgit(
                dec_output=backbone_out,
                temperature=temperature,
                topk=topk,
                use_cfg=self.config_unconditional_prob,
                cfg_scale=self.config_scale,
                n_steps=4,
                noise_scale=maskgit_noise_scale,
                fixed_schedule_n_unmasked=fixed_schedule_n_unmasked,
                dynamic_cfg_scale=dynamic_cfg_scale,
                sampling_type=maskgit_sampling_type,
            )
        return gen_tokens

    def pad_audio_to_factor(self, audio, audio_len, samples_per_frame, downsampling_factor: int = 1):
        """
        Zero pad the end of the audio so that we do not have a partial end frame.
        The output will be zero-padded to have an integer number of frames of
        length `samples_per_frame * downsampling_factor`.

        Args:
            audio: input time-domain signal (B, T)
            audio_len: valid length for each example in the batch (B,)
            samples_per_frame: number of samples per frame
            downsampling_factor: how much each frame is downsampled in later processing

        Returns:
            padded_audio: Padded time-domain signal (B, T')
            padded_len: Adjusted valid lengths (B,)
        """
        with fp32_precision():
            total_factor = samples_per_frame * downsampling_factor
            padded_len = total_factor * torch.ceil(audio_len / total_factor).int()
            max_len = padded_len.max().int().item()
            num_padding = max_len - audio.shape[1]
            padded_audio = F.pad(audio, (0, num_padding))
        return padded_audio, padded_len

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            logging.info(f"Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            super().load_state_dict(model_dict, strict=False)
