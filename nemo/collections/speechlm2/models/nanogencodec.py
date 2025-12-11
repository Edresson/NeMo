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
from transformers import AutoConfig, AutoModel, DynamicCache

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.tts.modules.audio_codec_modules import FiniteScalarQuantizer

from nemo.collections.audio.parts.utils.resampling import resample
from nemo.core.classes.module import NeuralModule
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.models.duplex_s2s_model import replace_control_speech_codes, tokens_to_str
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.asr_cer_wer import Intelligibility
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.metrics.secs import SECS
from nemo.collections.speechlm2.parts.metrics.token_accuracy import TokenAccuracy
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import (
    load_checkpoint,
    load_pretrained_hf,
    set_model_dict_for_partial_init,
    setup_audio_codec,
    setup_speech_encoder,
)

from nemo.collections.asr.models import ASRModel

from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging

from nemo.collections.tts.modules import transformer_2501


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
        return self.codec.vector_quantizer.codebook_size

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


class NanoGenCodec(LightningModule, HFHubMixin):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to NanoGenCodec as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        # convert dict to config
        cfg = DictConfig(cfg)
        self.cfg = cfg.model
        self.target_sample_rate = cfg.data.target_sample_rate
        self.source_sample_rate = cfg.data.source_sample_rate

        self.validation_save_path = os.path.join(cfg.exp_manager.explicit_log_dir, "validation_logs")

        # general configs
        self.use_speaker_embedding = self.cfg.get("use_speaker_embedding", False)
        self.parallel_codebook_loss_scale = self.cfg.get("parallel_codebook_loss_scale", 1.0)
        self.cfg_unconditional_prob = self.cfg.get('cfg_unconditional_prob', 0.0)
        self.cfg_scale = self.cfg.get('cfg_scale', None)
        self.use_local_transformer = self.cfg.get('use_local_transformer', False)
        self.local_transformer_type = self.cfg.get('local_transformer_type', "ar")
        self.local_transformer_loss_scale = self.cfg.get('local_transformer_loss_scale', 1.0)

        # ratio between the the codec frame rate and the decoder frame rate
        self.downsampling_factor = self.cfg.get('downsampling_factor', 1)
        self.frame_stacking_factor = self.cfg.get('frame_stacking_factor', 1)
        self.use_text_loss = self.cfg.get('use_text_loss', False)

        # codec configs
        setup_audio_codec(self)
        if self.frame_stacking_factor > 1:
            self.audio_codec = GroupedCodec(self.audio_codec, self.frame_stacking_factor)
            self._codebook_size = self.audio_codec._codebook_size
            self._num_codebooks = self.audio_codec._num_codebooks
        else:
            self._codebook_size = self.audio_codec.vector_quantizer.codebook_size
            self._num_codebooks = self.audio_codec.vector_quantizer.num_groups

        # compute target fps
        self.target_fps = self.target_sample_rate / self.audio_codec.samples_per_frame
        self.target_samples_per_frame = int(self.target_sample_rate//self.target_fps)

        # compute source fps
        self.source_fps = self.source_sample_rate / (
            self.source_sample_rate * cfg.data.frame_length
        )  # conver frame rate in fps
        self.source_samples_per_frame = int(self.source_sample_rate//self.source_fps)

        # Load tokenizer
        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        # set custom bos and eos
        self.tokenizer.bos_token = self.cfg.get("bos_token", '<|im_start|>')
        self.tokenizer.eos_token = self.cfg.get("eos_token", '<|im_end|>')
        # self.tokenizer.pad_token = self.cfg.get("pad_token", '<SPECIAL_12>')

        # Instanciate a from scratch model
        if self.cfg.get("backbone_type", None):
            backbone_config = AutoConfig.for_model(
                self.cfg.backbone_type,
                **(
                    OmegaConf.to_container(self.cfg.backbone_config, resolve=True)
                    if self.cfg.backbone_config
                    else {}
                ),
            )
            llm = AutoModel.from_config(backbone_config)
            self.decoder = llm  # fetch PretrainedBaseModel from model "ForCausalLM"
        else:
            # Load pretrained backbone from huggingface
            llm = load_pretrained_hf(self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights).train()
            self.decoder = llm.model  # fetch PretrainedBaseModel from model "ForCausalLM"

        # Note: we have to "move out" the token embedding outside of LLM to avoid
        #       messing up FSDP/TP hooks.
        # self.embed_text_tokens = self.decoder.embed_tokens
        del self.decoder.embed_tokens

        # audio embeddings
        audio_embeddings = []
        for _ in range(self._num_codebooks * self.downsampling_factor):
            audio_embeddings.append(nn.Embedding(self.speech_vocab_size, self.decoder.config.hidden_size))
        self.audio_embeddings = nn.ModuleList(audio_embeddings)

        # audio head
        self.final_proj = nn.Linear(self.decoder.config.hidden_size, self._num_codebooks * self.speech_vocab_size * self.downsampling_factor)

        # add loss
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

        if self.use_text_loss:
            self.text_head = llm.lm_head
            self.text_loss_scale = self.cfg.get('text_loss_scale', 1.0)

        # if use maskgit local transformer
        if self.use_local_transformer:
            local_transformer_hidden_dim = self.cfg.get('local_transformer_hidden_dim', 256)
            self.local_transformer_mask_token_id = self.speech_vocab_size - 1 # local transformer mask token

            # projection from model backbone to local transformer
            if local_transformer_hidden_dim != self.decoder.config.hidden_size:
                self.local_transformer_in_projection = nn.Linear(self.decoder.config.hidden_size, local_transformer_hidden_dim)
            else:
                self.local_transformer_in_projection = nn.Identity()

            self.local_transformer = transformer_2501.Transformer(
                n_layers=self.cfg.get('local_transformer_n_layers', 2),
                d_model=local_transformer_hidden_dim,
                d_ffn=local_transformer_hidden_dim*4,
                sa_n_heads=self.cfg.get('local_transformer_n_heads', 1),
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

        # init speaker encoder
        if self.use_speaker_embedding:
            self.speaker_encoder_model_name = self.cfg.get("speaker_encoder_model_name", 'titanet_large')
            self.max_speaker_reference_len = self.cfg.get("max_speaker_reference_len", 5)
            # setup speaker encoder
            self.setup_speaker_encoder()
            # speaker encoder projection
            self.speaker_encoder_emb_projection = nn.Linear(self.cfg.get("speaker_embedding_dim", 192), self.decoder.config.hidden_size)

        # Load the pretrained streaming ASR model and copy its parameters into the audio perception module.
        self.llm = self.decoder # set self.llm as self.decoder to allow the function get the the hidden size
        setup_speech_encoder(self, pretrained_weights=True)
        del self.llm

        self.use_fsq = self.cfg.get("fsq_quantizer_levels", None) is not None
        if self.use_fsq:
            bottleneck_dim = len(self.cfg.fsq_quantizer_levels)
            # Bottleneck projection → Quantizer → Projection back
            self.quantizer_bottleneck = nn.Linear(self.decoder.config.hidden_size, bottleneck_dim)
            self.vector_quantizer = FiniteScalarQuantizer(self.cfg.fsq_quantizer_levels)
            self.quantizer_projection = nn.Linear(bottleneck_dim, self.decoder.config.hidden_size)

        # cached for quicker audio decoding
        self.register_buffer(
            "_control_codes",
            torch.tensor([self.speech_bos_id, self.speech_eos_id, self.speech_delay_id], device=self.device),
        )
        self._use_fsdp = False
        self._use_tp = False

    def setup_speaker_encoder(self):
        with fp32_precision():
            self.speaker_encoder = EncDecSpeakerLabelModel.from_pretrained(model_name=self.speaker_encoder_model_name)

        # freeze the pretrained speaker encoder
        self.speaker_encoder.eval()
        self.speaker_encoder.freeze()

        for p in self.speaker_encoder.parameters():
            p.requires_grad = False

    def restore_from_pretrained_checkpoint(self, checkpoint_path):
        """
        Loads model weights a pretrained checkpoint file, supporting partial loading from safetensor and PyTorch formats.

        Args:
            checkpoint_path (str): Path to checkpoint file.

        Returns:
            None. The model is updated in-place.
        """
        if checkpoint_path is not None:
            checkpoint_state = load_checkpoint(checkpoint_path)
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.state_dict())

            if self.cfg.get("rescale_pretrained_weights", None):
                checkpoint_state = rescale_state_dict(
                    checkpoint_state, first_n_layers=self.cfg.get("rescale_first_n_layers", None)
                )

            self.load_state_dict(checkpoint_state, strict=True)

    @property
    def speech_vocab_size(self):
        """Return the size of the audio codec codebook including extra speech BOS and EOS tokens."""
        if self.use_local_transformer and self.local_transformer_type == "nar": # add extra token for mask
            return self._codebook_size + 4
        return self._codebook_size + 3

    @property
    def speech_bos_id(self) -> int:
        """Indicates start of utterance generation (not start of inference!)."""
        if self.cfg.get("custom_speech_bos_id", None):
            return self.cfg.get("custom_speech_bos_id")
        return self._codebook_size

    @property
    def speech_eos_id(self) -> int:
        """Indicates end of utterance generation."""
        if self.cfg.get("custom_speech_eos_id", None):
            return self.cfg.get("custom_speech_eos_id")
        return self._codebook_size + 1

    @property
    def speech_delay_id(self) -> int:
        """Indicates start of inference (the very first frame)."""
        if self.cfg.get("custom_speech_delay_id", None):
            return self.cfg.get("custom_speech_delay_id")
        return self._codebook_size + 2

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
    def text_pad_id(self) -> int:
        """
        Text pad ID is used as a 'blank' for frames when the model is not speaking
        and for frames where the model is speaking but has already predicted the
        entire text channel's content.

        Example:

            flow:         |---user---||-------assistant--------||-user-|
            text channel:  0000000000  1xxxxxxx0000000000000002  000000

        Where 0 indicates PAD ID, 1 indicates BOS ID, 2 indacates EOS ID,
        and x indicates tokens corresponding to actual text

        """
        return get_pad_id(self.tokenizer)

    def get_speaker_embedding(self, audio, audio_len, sr):
        # limit max audio len to avoid memory waste
        audio = audio[:, : int(self.max_speaker_reference_len*sr)]
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            with torch.no_grad():
                model_sr = self.speaker_encoder._cfg.train_ds.get('sample_rate', 16000)
                audio_resampled = resample(audio, sr, model_sr)
                audio_len_resampled = audio_len * (model_sr / sr )
                _, g = self.speaker_encoder(input_signal=audio_resampled, input_signal_length=audio_len_resampled.long())
                g = g.unsqueeze(1)
        return g

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
        input_embeds: Tensor,
        cache=None,
        seq_mask=None,
        force_disable_cfg=False,
    ) -> dict[str, Tensor]:
        """
        Separated text and speech prediction:
            - Speech prediction is achieved by a independent AR decoder based on last_hidden_state + audio tokens
            - For KV-cache:
                (1) llm cache depends on input cache is None or Not
                (2) speech_generation cache relys on reset_input_and_kv_cache function.
        """

        if self.cfg_unconditional_prob and not force_disable_cfg:
            if self.training:
                # if training drop the "text" conditioning in a percentage of batch
                if torch.rand(1).item() < self.cfg_unconditional_prob:
                    # make the whole batch zeros to the unconditional model
                    input_embeds = torch.zeros_like(input_embeds)
            elif self.cfg_scale is not None and not self.training:
                # if inference or evaluation create a zero tensor for decoder input and concatenate it to compute unconditional logits
                input_embeds_zeros = torch.zeros_like(input_embeds)
                input_embeds = torch.cat([input_embeds, input_embeds_zeros], dim=0)
                # duplicate mask to match the new shape
                if seq_mask is not None:
                    seq_mask = torch.cat([seq_mask, seq_mask], dim=0)

        out = self.decoder(
            inputs_embeds=input_embeds,
            attention_mask=seq_mask,
            past_key_values=cache, use_cache=cache is not None, return_dict=True
        )
        B, T = input_embeds.shape[:2]

        # get logits
        logits = self.final_proj(out.last_hidden_state)  # (B, T', num_codebooks * _codebook_size)

        text_logits = None
        if self.use_text_loss:
            text_logits = self.text_head(out.last_hidden_state)

        # if using cfg and it is in inference or evaluation mix unconditional and coditional logits
        if self.cfg_scale is not None and self.cfg_unconditional_prob and not self.training and not force_disable_cfg:
            batch_size = logits.size(0) // 2
            cond_logits = logits[:batch_size]
            uncond_logits = logits[batch_size:]
            logits = (1 - self.cfg_scale) * uncond_logits + self.cfg_scale * cond_logits

            if self.use_text_loss:
                text_cond_logits = text_logits[:batch_size]
                text_uncond_logits = text_logits[batch_size:]
                text_logits = (1 - self.cfg_scale) * text_uncond_logits + self.cfg_scale * text_cond_logits

        ans = {
            "logits": logits,
            "backbone_out": out.last_hidden_state,
            "text_logits": text_logits,
        }
        if cache is not None:
            ans["cache"] = out["past_key_values"]



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

    def prepare_inputs(self, batch: dict):
        """
        Similar to DuplexS2SModel.prepare_inputs, with following changes:
            (1) Add 'input_audio_tokens' and 'seq_mask' in return value for TransformerARSpeechDecoder
            (2) Remove audio codec embedding from 'input_embeds'
        """
        # check if audios has the same batch size
        assert batch["input_text_tokens"].size(0) == batch["target_audio"].size(0)
        # EARTTS dataloader add as default a eos token on the first text and a silence on audio, ignore it
        batch["target_audio"] = batch["target_audio"][:, self.target_samples_per_frame:]
        batch["input_text_tokens"] = batch["input_text_tokens"][:, 1:]
        
        input_text_tokens = batch["input_text_tokens"]

        # encoder embedding
        target_audio_asr_sr = resample(batch["target_audio"], self.target_sample_rate, self.cfg.get("asr_sample_rate", 16000))
        target_audio_lens_asr_sr = (batch["target_audio_lens"] / self.target_sample_rate * self.cfg.get("asr_sample_rate", 16000)).to(torch.long)
        encoded, encoded_len = self.perception(
            input_signal=target_audio_asr_sr, input_signal_length=target_audio_lens_asr_sr
        )

        if self.use_fsq:
            # Apply quantization: Linear → FSQ → Linear
            z = self.quantizer_bottleneck(encoded)
            # keep finite scalar quantization on FP32
            with fp32_precision():
                z_q, _ = self.vector_quantizer(inputs=z.transpose(1, 2), input_len=encoded_len)
            # make sure that it is runing in the right dtype
            z_q = z_q.transpose(1, 2).to(z.dtype)
            encoded = self.quantizer_projection(z_q)

        # extract target audio codes
        with fp32_precision(), torch.no_grad():
            target_codes, target_codes_lens = self.audio_codec.encode(
                audio=batch["target_audio"], audio_len=batch["target_audio_lens"]
            )
            target_codes = target_codes.transpose(1, 2)  # (B, K, T) -> (B, T, K)

        # Add and delay/BOS token to make sure that the first audio token is always a fixed token
        target_codes = torch.cat(
            [
                torch.full(
                    [target_codes.shape[0], 1 * self.downsampling_factor, target_codes.shape[-1]],
                    fill_value=self.speech_delay_id,
                    device=self.device,
                    dtype=torch.long,
                ),
                target_codes[:, :-  1 * self.downsampling_factor],
            ],
            dim=1,
        )

        # pad audio tokens to be divisible by the self.downsampling_factor
        # ToDo pad audio instead
        if self.downsampling_factor > 1:
            # pad tokens with speech_delay_token
            target_codes = self.pad_audio_codes_to_factor(target_codes, self.downsampling_factor, pad_token=self.speech_delay_id)
            # increase lens to the new size
            with fp32_precision():
                target_codes_lens = (target_codes_lens * (target_codes.size(1) / target_codes_lens.max())).to(target_codes_lens.dtype)


        with fp32_precision():
            target_len = target_codes.shape[1]

            # Pad or truncate sequence variables
            def pad_or_truncate(x, pad_value=0):
                if x.dim() == 2:  # [B, T]
                    L = x.shape[1]
                    if L < target_len:
                        return F.pad(x, (0, target_len - L), value=pad_value)
                    else:
                        return x[:, :target_len]
                return x  # leave others for now

            input_text_tokens = pad_or_truncate(input_text_tokens, pad_value=self.text_pad_id)

            if (tl := target_codes.shape[1]) != (sl := encoded.shape[1]):
                if tl < sl:
                    diff = sl - tl
                    encoded = encoded[:, :tl]
                    torch.clamp_(encoded_len, max=tl)
                else:
                    diff = tl - sl
                    target_codes = target_codes[:, :sl]
                    torch.clamp_(target_codes_lens, max=sl)

        # target codes to embeddings
        target_audio_emb, target_audio_emb_lens = self.embed_audio_tokens(
            target_codes, target_codes_lens
        )

        # create sequence mask
        seq_mask = get_mask_from_lengths(target_audio_emb_lens)

        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (target_codes.shape[1] - 1) % tp_world_size) != 0:
                target_codes = target_codes[:, :-remainder]
                encoded = encoded[:, :-remainder]
                target_audio_emb = target_audio_emb[:, :-remainder]
                seq_mask = seq_mask[:, :-remainder]
                encoded = encoded[:, :-remainder]

        # shift target codes embeddings for the autoregressive training
        audio_labels = target_codes[:, 1:] # (B, T-1, K)
        target_audio_emb = target_audio_emb[:, :-1] # (B, T-1, K)
        seq_mask = seq_mask[:, :-1]
        encoded = encoded[:, :-1]

        # Add sencoder embeddings
        input_embeds = encoded

        # Add shifted target codes embeddings
        input_embeds.add_(target_audio_emb)

        if self.use_speaker_embedding:
            # Extract speaker embedding
            audio_prompt = batch["audio_prompt"]
            audio_prompt_lens = batch["audio_prompt_lens"]
            speaker_encoder_emb = self.get_speaker_embedding(
                audio_prompt, audio_prompt_lens, self.target_sample_rate
            ).to(target_audio_emb.dtype)  # [B, D]

            speaker_embedding_projected = self.speaker_encoder_emb_projection(speaker_encoder_emb) # [B, 1, D]
            # Repeat along time dimension T to match input_embeds shape [B, T, D]
            speaker_embedding_expanded = speaker_embedding_projected.expand(-1, input_embeds.size(1), -1)  # [B, T, D]

            # Add to input_embeds
            input_embeds.add_(speaker_embedding_expanded)

        # create loss scale mask by copying seq_mask to include mask sequence
        loss_scale = seq_mask.clone().float()

        # debug samples:
        if (
            self.cfg.get("debug_dataloader_audios_path", None)
            and self.training
        ):


            def write_wave(one_audio_signal, file_name, sr=None):
                import numpy as np
                import soundfile as sf

                one_audio_signal = one_audio_signal.cpu().numpy()
                one_audio_signal = one_audio_signal.astype(np.float32)
                if sr is None:
                    sr = self.target_sample_rate
                # one_audio_signal = np.clip(one_audio_signal, -1.0, 1.0)
                sf.write(file_name, one_audio_signal, sr)

            def count_initial_pad_tokens(text_labels: torch.Tensor, text_pad_id: int) -> torch.Tensor:
                """
                Count the number of sequential text_pad_id tokens at the start of each sequence.

                Args:
                    text_labels: Tensor of shape [B, T] (token sequences).
                    text_pad_id: The pad token ID to count.

                Returns:
                    Tensor of shape [B,] with counts of initial pad tokens per sequence.
                """
                B, T = text_labels.shape
                is_pad = (text_labels == text_pad_id).to(torch.int32)  # [B, T]

                # Compute where pad sequence breaks using cumulative product
                mask = torch.cumprod(is_pad, dim=1)  # [B, T] will become 0 after first non-pad

                return mask.sum(dim=1)  # [B,]

            # encode and decode the audio
            with fp32_precision(), torch.no_grad():
                lengths = torch.tensor([batch["target_audio"].shape[1]] * batch["target_audio"].shape[0]).to(
                    self.audio_codec.device
                )
                reconstructed_audio_from_wav, _ = self.audio_codec(audio=batch["target_audio"], audio_len=lengths)
                # reconstruct wav
                audio_labels_ = replace_control_speech_codes(audio_labels, self._control_codes)
                with fp32_precision(), torch.no_grad():
                    lengths = torch.tensor([audio_labels_.shape[1]] * audio_labels_.shape[0]).to(
                        self.audio_codec.device
                    )
                    reconstructed_audio_from_tokens, _ = self.audio_codec.decode(
                        tokens=audio_labels_.transpose(1, 2), tokens_len=lengths
                    )
                # remove special tokens from labels
                audio_labels = replace_control_speech_codes(audio_labels, self._control_codes).transpose(1, 2)
                codec_latent = self.audio_codec.dequantize(audio_labels, target_codes_lens - 1 if self.downsampling_factor <= 1 else target_codes_lens).detach().transpose(1, 2)
                with fp32_precision(), torch.no_grad():
                    reconstructed_audio_from_codec_latent, _ = self.audio_codec.decode_audio(
                        inputs=codec_latent.transpose(1, 2), input_len=target_codes_lens - 1 if self.downsampling_factor <= 1 else target_codes_lens
                    )

            for i in range(audio_labels_.shape[0]):
                write_wave(
                    batch["target_audio"][i],
                    os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"target_audio_{i}.wav"),
                    sr=self.target_sample_rate,
                )
                write_wave(
                    batch["audio_prompt"][i],
                    os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"speaker_ref_{i}.wav"),
                    sr=self.target_sample_rate,
                )
                write_wave(
                    batch["source_audio"][i],
                    os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"source_audio_{i}.wav"),
                    sr=self.source_sample_rate,
                )

                write_wave(
                    reconstructed_audio_from_tokens[i],
                    os.path.join(
                        self.cfg.get("debug_dataloader_audios_path"), f"target_audio_reconstructed_from_tokens_{i}.wav"
                    ),
                    sr=self.target_sample_rate,
                )

                write_wave(
                    reconstructed_audio_from_wav[i],
                    os.path.join(
                        self.cfg.get("debug_dataloader_audios_path"),
                        f"target_audio_reconstructed_from_waveform_{i}.wav",
                    ),
                    sr=self.target_sample_rate,
                )
                
                write_wave(
                    reconstructed_audio_from_codec_latent[i],
                    os.path.join(
                        self.cfg.get("debug_dataloader_audios_path"), f"target_audio_reconstructed_from_codec_latent_{i}.wav"
                    ),
                    sr=self.target_sample_rate,
                )

            if audio_labels_.shape[0] > 1:
                exit()

        return {
            "input_embeds": input_embeds,
            "input_lens": target_codes_lens - 1 if self.downsampling_factor <= 1 else target_codes_lens,
            "output_lens": target_codes_lens - 1 if self.downsampling_factor <= 1 else target_codes_lens,
            "text_tokens": input_text_tokens,
            "audio_labels": audio_labels,
            "seq_mask": seq_mask,
            "loss_scale": loss_scale,
        }


    def compute_loss(self, logits, audio_codes, audio_codes_lens, mask_tokens_mask=None, loss_scale=None):
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
            audio_codes = audio_codes.transpose(1, 2)
            if loss_scale is None:
                loss_mask = get_mask_from_lengths(audio_codes_lens, pad_to_factor=self.downsampling_factor)
            else:
                loss_mask = loss_scale

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

    def training_step(self, batch: dict, batch_idx: int):

        for m in (self.decoder, self.audio_embeddings, self.final_proj):
            if is_frozen(m):
                m.eval()

        if self.use_speaker_embedding:
            for m in (self.speaker_encoder_emb_projection, self.speaker_encoder):
                if is_frozen(m):
                    m.eval()

        if self.use_local_transformer:
            for m in (self.local_transformer_in_projection, self.local_transformer):
                if is_frozen(m):
                    m.eval()

        inputs = self.prepare_inputs(batch)

        forward_outputs = self(
            inputs["input_embeds"],
            seq_mask=inputs["seq_mask"]
        )

        codebook_loss, loss_mask = self.compute_loss(forward_outputs["logits"], inputs["audio_labels"],  inputs["output_lens"], loss_scale=inputs["loss_scale"])

        # local transformer
        local_transformer_logits = None
        if self.use_local_transformer:
            if self.local_transformer_type == "ar":
                # autoregressive
                local_transformer_logits = self.compute_local_transformer_logits(forward_outputs["backbone_out"], inputs["audio_labels"], targets_offset_by_one=False)
                local_transformer_loss, _ = self.compute_loss(local_transformer_logits, inputs["audio_labels"], inputs["output_lens"], loss_scale=inputs["loss_scale"])
            else:
                # randomly replace some positions with MASK_TOKEN
                audio_codes_masked, mask_tokens_mask = self.maskgit_apply_random_mask(inputs["audio_labels"])
                local_transformer_logits = self.compute_local_transformer_logits(forward_outputs["backbone_out"], inputs["audio_labels"], targets_offset_by_one=True)
                local_transformer_loss, _ =  self.compute_loss(local_transformer_logits, inputs["audio_labels"], inputs["output_lens"], mask_tokens_mask, loss_scale=inputs["loss_scale"])
        else:
            local_transformer_loss = torch.tensor(0.0, device=self.device)

        loss = codebook_loss * self.parallel_codebook_loss_scale + local_transformer_loss * self.local_transformer_loss_scale
        num_frames = inputs["input_lens"].sum()
        B, T = inputs["input_embeds"].shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "codebook_loss": codebook_loss,
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "padding_ratio": num_frames / (B * T),
        }

        # add text loss
        if self.use_text_loss:
            with loss_parallel():
                text_loss = (
                    torch.nn.functional.cross_entropy(
                        forward_outputs["text_logits"].flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                        inputs["text_tokens"].flatten(0, 1),
                        reduction="sum",
                    )
                    / num_frames
                )
                ans["loss"] = ans["loss"] + (text_loss * self.text_loss_scale)
                ans["text_loss"] = text_loss

        self.log_dict(ans, on_step=True)
        return ans

    def on_train_epoch_start(self) -> None:
        setup_audio_codec(self)  # potentially reloads the audio codec to make sure it's in fp32
        # Replace audio_codec with GroupedCodec to enable frame stacking (downsampling).
        # The isinstance check prevents infinite recursion, since setup_audio_codec may already return a GroupedCodec when the codec is set to fp32.
        if self.frame_stacking_factor > 1 and not isinstance(self.audio_codec, GroupedCodec):
            self.audio_codec = GroupedCodec(self.audio_codec, self.frame_stacking_factor)

        if self.use_speaker_embedding:
            self.setup_speaker_encoder()  # potentially reloads the speaker encoder to make sure it's in fp32

    def on_validation_epoch_start(self) -> None:
        self.on_train_epoch_start()
        self.results_logger = ResultsLogger(self.validation_save_path).reset()
        self.asr_bleu = ASRBLEU(self.cfg.scoring_asr).reset()
        self.intelligibility = Intelligibility(self.cfg.scoring_asr, reuse_asr_hyps=True).reset()
        self.secs = SECS(self.cfg.get("scoring_se", "titanet_large")).reset()

    def on_validation_epoch_end(self, prefix="val") -> None:
        asr_bleu = self.asr_bleu.compute()
        for k, m in asr_bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        cer_wer = self.intelligibility.compute()
        for k, m in cer_wer.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        secs = self.secs.compute()
        for k, m in secs.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

    def get_teacher_force_inference_audio(self, batch):
        inputs = self.prepare_inputs(batch)

        forward_outputs = self(
            inputs["input_embeds"],
            seq_mask=inputs["seq_mask"],
            force_disable_cfg=True,
        )

        # local transformer
        local_transformer_logits = None
        if self.use_local_transformer:
            if self.local_transformer_type == "ar":
                # autoregressive
                local_transformer_logits = self.compute_local_transformer_logits(forward_outputs["backbone_out"], inputs["audio_labels"], targets_offset_by_one=False)
            else:
                # randomly replace some positions with MASK_TOKEN
                audio_codes_masked, mask_tokens_mask = self.maskgit_apply_random_mask(inputs["audio_labels"])
                local_transformer_logits = self.compute_local_transformer_logits(forward_outputs["backbone_out"], inputs["audio_labels"], targets_offset_by_one=True)

            lengths = torch.tensor([inputs["audio_labels"].shape[1]] * inputs["audio_labels"].shape[0]).to(self.audio_codec.device)
            tf_audio_codes_pred = self.logits_to_audio_codes(local_transformer_logits, lengths).transpose(1, 2)
        else:
            lengths = torch.tensor([inputs["audio_labels"].shape[1]] * inputs["audio_labels"].shape[0]).to(self.audio_codec.device)
            tf_audio_codes_pred = self.logits_to_audio_codes(forward_outputs["logits"], lengths).transpose(1, 2)

        # decode audio
        tf_audio_codes_pred = replace_control_speech_codes(tf_audio_codes_pred, self._control_codes)
        with fp32_precision(), torch.no_grad():
            lengths = torch.tensor([tf_audio_codes_pred.shape[1]] * tf_audio_codes_pred.shape[0]).to(
                self.audio_codec.device
            )
            tf_audio_pred, _ = self.audio_codec.decode(
                tokens=tf_audio_codes_pred.transpose(1, 2), tokens_len=lengths
            )

        return tf_audio_pred

    def validation_step(self, batch: dict, batch_idx: int):

        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            # resample target audio
            target_audio_asr_sr = resample(dataset_batch["target_audio"], self.target_sample_rate, 16000)
            target_audio_asr_sr_lens = (dataset_batch["target_audio_lens"] / self.target_sample_rate * 16000).to(torch.long)

            results = self.offline_inference(
                target_audio_asr_sr,
                target_audio_asr_sr_lens,
                speaker_audio=dataset_batch["audio_prompt"],
                speaker_audio_lens=dataset_batch["audio_prompt_lens"],
            )

            results["tf_audio_pred"] = self.get_teacher_force_inference_audio(dataset_batch)

            with fp32_precision():  # resample is fragile to bfloat16 default dtype
                metric_audio_pred = results["audio"]
                metric_audio_pred_lens = results["audio_len"]

                # resample audio to the asr sampling rate
                metric_audio_pred = resample(metric_audio_pred, self.target_sample_rate, 16000)
                metric_audio_pred_lens = (metric_audio_pred_lens / self.target_sample_rate * 16000).to(torch.long)

                if self.cfg.get("use_GT_transcriptions_for_metrics", True):
                    # use target audio transcription for metrics
                    target_asr_texts = self.asr_bleu.asr.transcribe(
                        [
                            audio[:alen]
                            for audio, alen in zip(target_audio_asr_sr, target_audio_asr_sr_lens)
                        ],
                        batch_size=target_audio_asr_sr.shape[0],
                        verbose=False,
                    )
                    metric_text = [asr_hyp.text for asr_hyp in target_asr_texts]
                else:
                    metric_text = dataset_batch["target_texts"]
                    
                asr_hyps = self.asr_bleu.update(
                    name=name,
                    refs=metric_text,
                    pred_audio=metric_audio_pred,
                    pred_audio_lens=metric_audio_pred_lens,
                )

                self.intelligibility.update(
                    name=name,
                    refs=metric_text,
                    pred_audio=metric_audio_pred,
                    pred_audio_lens=metric_audio_pred_lens,
                    asr_hyps=asr_hyps,
                )

                # add ground truth intelligibility metrics
                self.intelligibility.update(
                    name=name + "_gt",
                    refs=dataset_batch["target_texts"],
                    pred_audio=target_audio_asr_sr,
                    pred_audio_lens=target_audio_asr_sr_lens,
                    asr_hyps=(
                        metric_text if self.cfg.get("use_GT_transcriptions_for_metrics", True) else None
                    ),  # reuse GT transcription
                )

                self.secs.update(
                    name=name,
                    target_audio=target_audio_asr_sr,
                    target_audio_lens=target_audio_asr_sr_lens,
                    pred_audio=resample(results["audio"], self.target_sample_rate, 16000),
                    pred_audio_lens=(results["audio_len"] / self.target_sample_rate * 16000).to(torch.long),
                )

                self.results_logger.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    hyps=metric_text,
                    asr_hyps=asr_hyps,
                    samples_id=dataset_batch['sample_id'],
                    pred_audio=results["audio"],
                    pred_audio_tf=results["tf_audio_pred"],
                    pre_audio_trimmed=None,
                    reference_audio=dataset_batch["audio_prompt"].float(),
                    target_audio=dataset_batch["target_audio"].float(),
                    pred_audio_sr=self.target_sample_rate,
                    user_audio=dataset_batch["source_audio"],
                    user_audio_sr=self.source_sample_rate,
                    eou_pred=None,
                    fps=self.target_fps,
                    results=results if self.cfg.get("dump_tokens_text", False) else None,
                    tokenizer=self.tokenizer,
                )

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)


    def sample_codes_from_logits(self, all_code_logits_t, temperature=0.7, topk=80, unfinished_items={}, finished_items={}):
        # all_code_logits_t: (B, num_codebooks * num_tokens_per_codebook), logits at a given timestep
        all_preds = [[] for _ in range(self.downsampling_factor)]
        for ds_index in range(self.downsampling_factor):
            for idx in range(self._num_codebooks):
                si = (idx + self._num_codebooks * ds_index) * self.speech_vocab_size
                ei = si + self.speech_vocab_size
                codebook_logits = all_code_logits_t[:, si:ei]  # (B, num_tokens_per_codebook)
                for item_idx in unfinished_items:
                    codebook_logits[item_idx, self.speech_eos_id] = float('-inf')
                for item_idx in finished_items:
                    codebook_logits[item_idx, :] = float('-inf')
                    codebook_logits[item_idx, self.speech_eos_id] = 0.0
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
                use_cfg=self.cfg_unconditional_prob,
                cfg_scale=self.cfg_scale
            )

        else:
            gen_tokens = self.local_transformer_sample_maskgit(
                dec_output=backbone_out,
                temperature=temperature,
                topk=topk,
                use_cfg=self.cfg_unconditional_prob,
                cfg_scale=self.cfg_scale,
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

    @torch.no_grad()
    def offline_inference(
        self,
        input_audio: torch.Tensor,
        input_audio_lens: torch.Tensor,
        speaker_audio: torch.Tensor,
        speaker_audio_lens: torch.Tensor,
        decode_audio: bool = True,
        formatter: str = "",
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive prediction.

        Args:
            input_signal: a batch of waveforms with shape (B, T) with source sampling rate.
            input_signal_lens: example lengths as number of samples of shape (B,).
            decode_audio: bool, whether to decode audio codes to waveform.

        Returns:
            A dict with keys:
                * "text": generated text, de-tokenized to strings, properly skipping text_pad_id; list of length B.
                * "tokens_text": generated text tokens of shape (B, T2).
                * "tokens_audio": generated audio codes of shape (B, T2, K) where `K=num_codebooks`.
                * "tokens_len" output lengths as number of tokens of shape (B,).
                * "audio": generated waveform of shape (B, T3) (`decode_audio=True`).
                * "audio_len" output lengths as number of waveform samples of shape (B,) (when `decode_audio=True`).
        """

        input_audio, input_audio_lens = self.pad_audio_to_factor(input_audio, input_audio_lens, self.source_samples_per_frame, self.downsampling_factor)

        encoded, encoded_len = self.perception(
            input_signal=input_audio, input_signal_length=input_audio_lens
        )

        if self.use_fsq:
            # Apply quantization: Linear → FSQ → Linear
            z = self.quantizer_bottleneck(encoded)
            # keep finite scalar quantization on FP32
            with fp32_precision():
                z_q, _ = self.vector_quantizer(inputs=z.transpose(1, 2), input_len=encoded_len)
            # make sure that it is runing in the right dtype
            z_q = z_q.transpose(1, 2).to(z.dtype)
            encoded = self.quantizer_projection(z_q)

        # create the speaker embedding to reuse in the autoregressive loop
        if self.use_speaker_embedding:
            speaker_emb = self.get_speaker_embedding(
                speaker_audio, speaker_audio_lens, self.target_sample_rate
            ).to(encoded.dtype).to(encoded.device)

            # project speaker embedding to match llm input size
            speaker_emb = self.speaker_encoder_emb_projection(speaker_emb)  # [B, 1, D]

        B, T_local, H = encoded.shape

        # Determine decoding length and pad if FSDP
        if self._use_fsdp:
            T_tensor = torch.tensor([T_local], device=encoded.device)
            dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
            T = int(T_tensor.item())
            if T > T_local:
                # pad source emb
                last_frame_source = encoded[:, T_local - 1: T_local, :]
                pad_source = last_frame_source.repeat(1, T - T_local, 1)
                encoded = torch.cat([encoded, pad_source], dim=1)
        else:
            T = T_local

        # Create empty tensor for store model input
        input_embeds = torch.zeros(B, T, H, device=self.device, dtype=encoded.dtype) # encoded.clone()

        # This cache is for self.decoder
        cache = DynamicCache()
        gen_audio_codes = torch.zeros(B, T, self._num_codebooks, self.downsampling_factor, device=self.device, dtype=torch.long)

        # Add source audio first frame to the model input
        input_embeds[:, 0] = encoded[:, 0]

        # first audio tokens are full with speech_delay_id
        first_audio_codes = torch.full(
            [B, 1 * self.downsampling_factor, self._num_codebooks],
            fill_value=self.speech_delay_id,
            device=self.device,
            dtype=torch.long,
        )

        # Add shifted target codes embeddings
        target_audio_emb = self.embed_audio_tokens(
            first_audio_codes
        )
        input_embeds[:, 0] += target_audio_emb[:, 0]

        if self.use_speaker_embedding:
            input_embeds[:, 0 : 1] += speaker_emb

        ans = self(
            input_embeds[:, :1],
            cache=cache,
            seq_mask=None,
        )
        if self.use_local_transformer:
            gen_audio_codes[:, 0] = self.local_transformer_sample_codes_from_logits(ans["backbone_out"][:, -1], temperature=self.cfg.get('temperature', 0.7), topk=self.cfg.get('topk', 80)) # (B, num_codebooks)
        else:
            gen_audio_codes[:, 0] = self.sample_codes_from_logits(ans["logits"][:, -1], temperature=self.cfg.get('temperature', 0.7), topk=self.cfg.get('topk', 80)) # (B, num_codebooks)

        speech_state = torch.zeros(B, device=self.device, dtype=torch.long)

        bos_indices = torch.full((B,), fill_value=-1, dtype=torch.long, device=gen_audio_codes.device)
        eos_indices = torch.full((B,), fill_value=-1, dtype=torch.long, device=gen_audio_codes.device)  # -1 means "not ended yet"
        # Autoregressive loop
        for t in range(1, T):

            # add source audio embedding          
            input_embeds[:, t] += encoded[:, t]

            # add audio tokens
            prev_audio_codes = gen_audio_codes[:, t - 1 : t, :] 
            # gen_audio_codes B, T=?, C=8, F=2
            # prev_audio_codes  B, C=8, F=2
            input_embeds[:, t] += self.embed_audio_tokens(
                prev_audio_codes.transpose(1, 2).reshape(prev_audio_codes.size(0), self._num_codebooks, -1).transpose(1, 2) # transpose(1, 2) to make the self._num_codebooks dimention as second and then collapse two last dim (T and self.downsampling_factor) then transpose it back
                # prev_audio_codes.reshape(prev_audio_codes.size(0), -1, self._num_codebooks) # reshape to handle self.downsampling_factor: Roy: double check
            )[:, -1]

            if self.use_speaker_embedding:
                input_embeds[:, t : t + 1] += speaker_emb

            ans = self(
                input_embeds[:, t : t + 1],
                cache=ans["cache"],
                seq_mask=None,
            )

            if self.use_local_transformer:
                gen_audio_codes[:, t] = self.local_transformer_sample_codes_from_logits(ans["backbone_out"][:, -1], temperature=self.cfg.get('temperature', 0.7), topk=self.cfg.get('topk', 80)) # (B, num_codebooks)
            else:
                gen_audio_codes[:, t] = self.sample_codes_from_logits(ans["logits"][:, -1], temperature=self.cfg.get('temperature', 0.7), topk=self.cfg.get('topk', 80)) # (B, num_codebooks)


        # Trim back to local length if padded
        if self._use_fsdp and T > T_local:
            gen_audio_codes = gen_audio_codes[:, :T_local]

        # expand lengths to match the new size
        with fp32_precision():
            tokens_audio_len = torch.ceil(encoded_len * self.downsampling_factor).to(encoded_len.dtype)

        # gen_audio_codes B, T=?, C=8, F=2
        ans = {
            "tokens_audio": gen_audio_codes.transpose(1, 2).reshape(gen_audio_codes.size(0), self._num_codebooks, -1).transpose(1, 2),
            "tokens_text_len": encoded_len,
            "tokens_audio_len": tokens_audio_len,
        }

        if decode_audio:
            gen_audio_codes = replace_control_speech_codes(ans["tokens_audio"], self._control_codes)
            with fp32_precision(), torch.no_grad():
                predicted_audio, predicted_audio_lens = self.audio_codec.decode(
                    tokens=gen_audio_codes.transpose(1, 2), tokens_len=tokens_audio_len
                )
            ans["audio"] = predicted_audio
            ans["audio_len"] = predicted_audio_lens
            ans["trimmed_audio"] = None
            ans["trimmed_audio_len"] = None

        return ans

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        return configure_optimizers(self)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        return {
            "cls": dict,
            "inputs": [
                {"name": "source_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "source_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {"name": "target_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "target_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "target_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.tokenizer.vocab_size,
                },
            ],
        }

    def configure_model(self) -> None:
        # TODO(pzelasko): refactor into separate module re-usable across models
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.decoder
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            for transformer_block in llm.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: "
                            f"set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                parallelize_module(transformer_block, tp_mesh, plan)

            for m in (self.final_proj, self.audio_embeddings):
                parallelize_module(
                    m,
                    tp_mesh,
                    ColwiseParallel(
                        input_layouts=Shard(1),
                        output_layouts=Shard(-1),
                        use_local_output=False,
                    ),
                )

            if self.use_local_transformer:
                for m in (self.local_transformer_in_projection, self.local_transformer):
                    parallelize_module(
                        m,
                        tp_mesh,
                        ColwiseParallel(
                            input_layouts=Shard(1),
                            output_layouts=Shard(-1),
                            use_local_output=False,
                        ),
                    )

            if self.use_speaker_embedding:
                for m in (self.speaker_encoder_emb_projection):
                    parallelize_module(
                        m,
                        tp_mesh,
                        ColwiseParallel(
                            input_layouts=Shard(1),
                            output_layouts=Shard(-1),
                            use_local_output=False,
                        ),
                    )


        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1
            self._use_fsdp = True

            fsdp_config = {"mesh": dp_mesh}

            for idx, layer in enumerate(llm.layers):
                llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.decoder = fully_shard(self.decoder, **fsdp_config)
            self.final_proj = fully_shard(self.final_proj, **fsdp_config)
            for idx in range(self._num_codebooks):
                self.audio_embeddings[idx] = fully_shard(self.audio_embeddings[idx], **fsdp_config)
                
            if self.use_local_transformer:
                self.local_transformer = fully_shard(self.local_transformer, **fsdp_config)
                self.local_transformer_in_projection = fully_shard(self.local_transformer_in_projection, **fsdp_config)

            if self.use_speaker_embedding:
                self.speaker_encoder_emb_projection = fully_shard(self.speaker_encoder_emb_projection, **fsdp_config)

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            logging.info(f"Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            super().load_state_dict(model_dict, strict=False)
