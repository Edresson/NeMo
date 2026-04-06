import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from nemo.collections.asr.models import ASRModel
from nemo.collections.speechlm2.parts.pretrained import setup_speech_encoder
from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.optim_setup import is_frozen
from nemo.collections.tts.modules.audio_codec_modules import FiniteScalarQuantizer
from nemo.collections.speechlm2.models.duplex_ear_tts import (
    RVQEARTTSModel,
    DuplexEARTTS,
    setup_audio_codec,
    replace_control_speech_codes,
    ensures_target_precision,
)
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from nemo.collections.speechlm2.models import SALM, SALMWithAsrDecoder
import gc

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import MimiConfig
from transformers.models.mimi.modeling_mimi import MimiEncoder, MimiTransformerModel, MimiConv1d, MimiConvTranspose1d, MimiDecoder
from nemo.core.classes.module import NeuralModule
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths

from nemo.collections.common.parts.utils import ClampActivation
from nemo.collections.tts.modules.audio_codec_modules import CodecActivation, CausalConv1dNorm




from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from transformers import MimiModel

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int = 1152,
        layer_scale_init_value: float = 1.0,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        # self.dwconv = CausalConv1D(dim, dim, kernel_size=7, padding=None, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x

class TemporalSmoothingHead(nn.Module):
    def __init__(self, in_channels=882, kernel_size=5, out_kernel_size=3, pad_mode="zeros", output_activation="clamp", activation="half_snake"):
        super().__init__()
        
        self.pre_conv = CausalConv1dNorm(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, pad_mode=pad_mode)
        self.pre_activation = CodecActivation(activation, channels=in_channels)
        self.post_conv = CausalConv1dNorm(in_channels=in_channels, out_channels=in_channels, kernel_size=out_kernel_size, pad_mode=pad_mode)
        if output_activation == "tanh":
            self.out_activation = nn.Tanh()
        elif output_activation == "clamp":
            self.out_activation = ClampActivation()

    def forward(self, x, x_len):
        out = x.transpose(1, 2)
        out = self.pre_conv(inputs=out, input_len=x_len)
        out = self.pre_activation(out)
        # [B, 1, T_audio]
        out = self.post_conv(inputs=out, input_len=x_len)
        out = self.out_activation(out)
        return out.transpose(1, 2)

from contextlib import contextmanager
@contextmanager
def default_precision(dtype=torch.float32):
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(default_dtype)


class ReshapeTransformerEncoder(NeuralModule):
    """
    Transformer Audio encoder.

    Args:
        output_dim: Dimension of encoder output.
    """

    def __init__(
        self,
        samples_per_frame: int,
        audio_proj_size: int = 1024, 
        output_dim: int = 32,
        n_layers: int = 8,
        d_model: int = 1024,
        d_ffn: int = 4096,
        is_causal: bool = True,
        sliding_window_size: int = 12,
        max_position_embeddings: int = 8000,
        rope_theta: float = 10000.0,
        attn_implementation: str = "eager",
    ):
        super().__init__()

        self.is_causal = is_causal
        self.samples_per_frame = samples_per_frame
        self.audio_proj_size = audio_proj_size
        self.output_dim = output_dim

        self.config = MimiConfig()
        self.config._attn_implementation = attn_implementation
        self.config.max_position_embeddings = max_position_embeddings
        self.config.rope_theta = rope_theta

        self.config.use_causal_conv = is_causal
        self.config.num_hidden_layers = n_layers
        self.config.intermediate_size = d_ffn
        self.config.hidden_size = d_model
        self.config.sliding_window = sliding_window_size
        self.layers = MimiTransformerModel(self.config)

        self.inp_projection_no_bias = nn.Linear(samples_per_frame, audio_proj_size, bias=False)
        self.inp_projection = nn.Linear(audio_proj_size, d_model)
        self.out_projection = nn.Linear(d_model, output_dim)

    def forward(self, audio, audio_len):
        encoded_len = audio_len
        B, T = audio.size()
        audio = audio.reshape(B, -1, self.samples_per_frame) # B, T, F, where 7 is the number of samples per frame that controls the frame rate
        with default_precision(torch.float32):
            encoded_len = (audio_len / self.samples_per_frame).long()

        if self.is_causal:
            mask = get_mask_from_lengths(encoded_len)
        else:
            # mask none does not apply causal mask
            mask = None

        out = self.inp_projection_no_bias(audio)
        out = self.inp_projection(out)

        out = self.layers(out, attention_mask=mask)[0]
        # out projection
        encoded = self.out_projection(out).transpose(1, 2)
        return encoded, encoded_len


class ReshapeTransformerDecoder(NeuralModule):
    """
    Transformer Audio Decoder.

    Args:
        input_dim: Dimension of encoder output.
    """

    def __init__(
        self,
        samples_per_frame: int,
        audio_proj_size: int = 1024, 
        input_dim: int = 32,
        n_layers: int = 8,
        d_model: int = 1024,
        d_ffn: int = 4096,
        is_causal: bool = True,
        sliding_window_size: int = 12,
        max_position_embeddings: int = 8000,
        rope_theta: float = 10000.0,
        attn_implementation: str = "eager",
        use_conv_pos: bool = False,
        num_pos_conv_blocks: int = 1,
        use_temporal_smoth_head: bool = False,
    ):
        super().__init__()

        self.samples_per_frame = samples_per_frame
        self.audio_proj_size = audio_proj_size
        self.is_causal = is_causal
        self.use_conv_pos = use_conv_pos
        self.use_temporal_smoth_head = use_temporal_smoth_head

        self.config = MimiConfig()
        self.config._attn_implementation = attn_implementation
        self.config.max_position_embeddings = max_position_embeddings
        self.config.rope_theta = rope_theta

        self.config.use_causal_conv = is_causal
        self.config.num_hidden_layers = n_layers
        self.config.intermediate_size = d_ffn
        self.config.hidden_size = d_model
        self.config.sliding_window = sliding_window_size
        self.layers = MimiTransformerModel(self.config)

        self.inp_projection = nn.Linear(input_dim, d_model)
        self.out_projection = nn.Linear(d_model, audio_proj_size)
        self.out_projection_no_bias = nn.Linear(audio_proj_size, samples_per_frame, bias=False)

        # add ConvNeXt based conv pos
        if self.use_conv_pos:
            self.conv_pos = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=d_model
                )
                for _ in range(num_pos_conv_blocks)
            ]
        )

        if self.use_temporal_smoth_head:
            self.temporal_smoth_head = TemporalSmoothingHead(d_model)

    def forward(self, inputs, input_len):
        if self.is_causal:
            mask = get_mask_from_lengths(input_len)
        else:
            # mask none does not apply causal mask
            mask = None

        encoded_len = input_len
        out = self.inp_projection(inputs.transpose(1, 2))
        out = self.layers(out, attention_mask=mask)[0]

        if self.use_conv_pos:
            out = out.transpose(1, 2)
            for conv_block in self.conv_pos:
                out = conv_block(out)
            out = out.transpose(1, 2)

        if self.use_temporal_smoth_head:
            out = self.temporal_smoth_head(out, input_len)

        out = self.out_projection(out)
        audio = self.out_projection_no_bias(out)

        # resample audio to size
        audio = audio.reshape(inputs.size(0), -1)
        audio_len = (input_len*self.samples_per_frame).int()
        return audio, audio_len


class MimiAudioEncoder(NeuralModule):
    def __init__(self, config, out_size=32):
        super().__init__()
        self.is_causal = config.use_causal_conv

        # get Mimi default config
        self.config = config
        self.config._attn_implementation = "eager"

        # define upsampling rate
        self.downsampling_rate = self.config.sampling_rate / self.config.frame_rate

        self.encoder = MimiEncoder(self.config)
        self.encoder_transformer = MimiTransformerModel(self.config)

        # extra downsample requeried because MiMiEncoder works in a different frame rate
        self.use_extra_downsample = self.encodec_frame_rate != self.config.frame_rate
        if self.use_extra_downsample:
            self.downsample = MimiConv1d(
                self.config,
                self.config.hidden_size,
                self.config.hidden_size,
                kernel_size=2 * int(self.encodec_frame_rate / self.config.frame_rate),
                stride=2,
                bias=False,
                pad_mode="replicate",
            )

        self.out_projection = MimiConv1d(
            self.config,
            self.config.hidden_size,
            out_size,
            kernel_size=1,
            stride=1,
            bias=False,
            pad_mode="replicate",
        )

    @property
    def encodec_frame_rate(self) -> int:
        hop_length = np.prod(self.config.upsampling_ratios)
        return math.ceil(self.config.sampling_rate / hop_length)

    def forward(self, input_signal, input_signal_length):
        audio = input_signal
        audio_len = input_signal_length

        if self.is_causal:
            mask = get_mask_from_lengths(audio_len)
        else:
            # mask none does not apply causal mask
            mask = None

        audio = audio.unsqueeze(1)
        embeddings = self.encoder(audio)
        embeddings = self.encoder_transformer(
            embeddings.transpose(1, 2), attention_mask=mask
        )[0].transpose(1, 2)

        if self.use_extra_downsample:
            embeddings = self.downsample(embeddings)

        embeddings = self.out_projection(embeddings)

        # compute output_len based on downsampling rate
        output_len = (audio_len / self.downsampling_rate).long()
        return embeddings.transpose(1, 2), output_len


class MimiAudioDecoder(NeuralModule):
    def __init__(self, input_size=32, sampling_rate=24000, upsampling_ratios=[8, 6, 5, 4], frame_rate=12.5, is_causal=True, hidden_size=512, sliding_window=250, num_transformer_layers=8):
        super().__init__()
        self.is_causal = is_causal

        # get Mimi default config
        self.config = MimiConfig()
        self.config._attn_implementation = "eager"

        # redefine configs based on nemo configs
        self.config.frame_rate = frame_rate
        self.config.sampling_rate = sampling_rate
        self.config.upsampling_ratios = upsampling_ratios
        self.config.use_causal_conv = is_causal
        self.config.hidden_size = hidden_size
        self.config.sliding_window = sliding_window
        self.config.num_hidden_layers = num_transformer_layers

        # define upsampling rate
        self.upsampling_rate = self.config.sampling_rate / self.config.frame_rate

        self.decoder_transformer = MimiTransformerModel(self.config)
        self.decoder = MimiDecoder(self.config)

        # extra upsampling requeried because MiMiEncoder works in a different frame rate
        self.use_extra_upsample = self.encodec_frame_rate != self.config.frame_rate
        if self.use_extra_upsample:
            self.upsample = MimiConvTranspose1d(
                self.config,
                self.config.hidden_size,
                self.config.hidden_size,
                kernel_size=2 * int(self.encodec_frame_rate / self.config.frame_rate),
                stride=2,
                bias=False,
                groups=self.config.upsample_groups,
            )

        self.in_projection = MimiConv1d(
            self.config,
            input_size,
            self.config.hidden_size,
            kernel_size=1,
            stride=1,
            bias=False,
            pad_mode="replicate",
        )

    @property
    def encodec_frame_rate(self) -> int:
        hop_length = np.prod(self.config.upsampling_ratios)
        return math.ceil(self.config.sampling_rate / hop_length)

    def forward(self, inputs, input_len, past_key_values=None, return_dict=None, return_past_key_values=False):
        if self.is_causal:
            mask = get_mask_from_lengths(input_len)
        else:
            # mask none does not apply causal mask
            mask = None

        embeddings = self.in_projection(inputs)
        if self.use_extra_upsample:
            embeddings = self.upsample(embeddings)

        decoder_outputs = self.decoder_transformer(
            embeddings.transpose(1, 2), attention_mask=mask, past_key_values=past_key_values, return_dict=return_dict
        )

        embeddings = decoder_outputs[0].transpose(1, 2)
        outputs = self.decoder(embeddings).squeeze(1)
        # compute output len based on the upsampling rate
        output_len = (input_len * self.upsampling_rate).long()
        if return_past_key_values:
            if return_dict:
                past_key_values = decoder_outputs.get("past_key_values")
            elif len(decoder_outputs) > 1:
                past_key_values = decoder_outputs[1]
            return outputs, past_key_values
        return outputs, output_len


class ImprovedFiniteScalarQuantizer(FiniteScalarQuantizer):
    """
    Improved Finite Scalar Quantization (iFSQ).

    Inherits from FiniteScalarQuantizer but replaces the standard tanh bounding
    with a distribution-matching scaled sigmoid to force a uniform distribution,
    maximizing codebook utilization.

    References:
        iFSQ Paper (https://arxiv.org/abs/2601.17124)
    """

    def __init__(self, num_levels: List[int], eps: float = 1e-3):
        super().__init__(num_levels=num_levels, eps=eps)

    def compress(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Apply iFSQ compression to the input to achieve uniform bin utilization."""
        output_scale = (self.num_levels - 1) / 2
        # scale down a bit to avoid rounding issues
        output_scale = output_scale * (1 - self.eps)
        # offset for even number of levels
        output_offset = torch.where(self.num_levels % 2 == 0, 0.5, 0.0)

        # Calculate the shift required to center even-numbered levels.
        # For iFSQ, the activation is y = 2 * sigmoid(1.6x) - 1.
        # The exact mathematical inverse to find the shift is x = logit((y + 1)/2) / 1.6
        if torch.any(self.num_levels % 2 == 0):
            y_target = output_offset / output_scale
            # Clamp safely to avoid infinities in logit
            y_target = torch.clamp(y_target, min=-1.0 + 1e-5, max=1.0 - 1e-5)
            input_shift = torch.logit((y_target + 1.0) / 2.0) / 1.6
        else:
            # If all levels are odd (e.g., [13, 13, 13, 13, 9]), no shift is needed.
            input_shift = torch.zeros_like(output_offset)

        # ---------------------------------------------------------------------
        # The Core iFSQ Improvement: Scaled Sigmoid instead of Tanh
        # ---------------------------------------------------------------------
        shifted_inputs = inputs + input_shift
        ifsq_activation = 2.0 * torch.sigmoid(1.6 * shifted_inputs) - 1.0

        output = output_scale * ifsq_activation - output_offset
        return output


class GenerativeCodecRVQEARTTSModel(RVQEARTTSModel):
    """
    Overrides RVQEARTTSModel to ignore textual inputs and prioritize
    the ASR/FSQ embedding for generative codec modeling.
    """

    def __init__(self, config, tokenizer=None):
        super().__init__(config, tokenizer)

        # Completely remove unused textual modules from the PyTorch registry
        # This saves VRAM and prevents FSDP gradient synchronization crashes
        if hasattr(self, 'embed_subword') and self.embed_subword is not None:
            del self.embed_subword
            self.embed_subword = None

        if hasattr(self, 'embed_context') and self.embed_context is not None:
            del self.embed_context
            self.embed_context = None

    def forward(self, *args, **kwargs):
        # Explicitly nullify text-based inputs to force the model
        # to rely entirely on `asr_speech_tokens_emb`
        kwargs['subword_ids'] = None
        kwargs['subword_mask'] = None
        kwargs['context_hidden_state'] = None

        return super().forward(*args, **kwargs)


class GenerativeCodecEARTTS(DuplexEARTTS):
    """
    Inherits from DuplexEARTTS to instantiate an ASR encoder and FSQ quantizer,
    bypassing text conditioning for a pure generative codec approach.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

        # Replace base TTS model with our text-ignoring class
        self.tts_model = GenerativeCodecRVQEARTTSModel(DictConfig(self.cfg.tts_config), tokenizer=self.tokenizer)
        # Replace codec and also load rvq embeddings
        setup_audio_codec(self)

        if self.cfg.get("use_pretrained_quantizer", False):
            # Load the massive full model to the CPU, NOT the GPU.
            # This prevents your VRAM from spiking during initialization.
            full_salm = SALM.from_pretrained(
                self.cfg.pretrained_quantizer_name_or_path, 
                map_location="cpu"
            )
            
            # 2. Extract the 600M perception module
            self.perception = full_salm.perception
            # 3. Sever the tie so the parent model doesn't hold onto the perception weights
            del full_salm.perception 
            # 4. Delete the massive parent model (the ~400M leftover params)
            del full_salm
            # 5. Force Python's Garbage Collector to immediately free the system RAM
            gc.collect()
            # Force PyTorch to release any cached memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Create the projection
            self.quantizer_projection = nn.Linear(self.perception.cfg.output_dim, self.tts_model.hidden_size)
            
        else:
            if self.cfg.get("use_mimi_encoder", False):
                mimi_model = MimiModel.from_pretrained("kyutai/mimi")
                pre_quant_hidden = mimi_model.config.hidden_size

                self.perception = MimiAudioEncoder(
                    out_size=pre_quant_hidden, 
                    config=mimi_model.config
                )

                state_dict_to_load = mimi_model.state_dict()

                # 1. Get the total number of keys in your custom model
                total_model_keys = len(self.perception.state_dict().keys())
                
                # 2. Load the weights
                load_result = self.perception.load_state_dict(state_dict_to_load, strict=False)
                missing_keys = load_result.missing_keys
                unexpected_keys = load_result.unexpected_keys
                
                # 3. Calculate how many keys were successfully restored
                restored_keys_count = total_model_keys - len(missing_keys)
                
                # 4. Print the diagnostic information
                print(f"--- Mimi Encoder Weight Loading ---")
                print(f"Total keys in self.perception: {total_model_keys}")
                print(f"Successfully restored keys: {restored_keys_count}")
                print(f"-----------------------------------")

                # Optional: Print warnings if crucial weights are missing
                if missing_keys:
                    print(f"Warning: {len(missing_keys)} Missing keys when loading Mimi encoder: {missing_keys[:5]}...")
                if unexpected_keys:
                    print(f"Warning: {len(unexpected_keys)} Unexpected keys when loading Mimi encoder: {unexpected_keys[:5]}...")

                self.cfg.asr_sample_rate = mimi_model.config.sampling_rate

                # Cleanup
                del state_dict_to_load
                del mimi_model
                gc.collect()
            else:
                # Temporarily mock self.llm so setup_speech_encoder works here
                self.llm = SimpleNamespace(config=SimpleNamespace(hidden_size=self.tts_model.hidden_size))
                # Setup the Speech Encoder (ASR model perception)
                setup_speech_encoder(self, pretrained_weights=True)
                pre_quant_hidden = self.tts_model.hidden_size

            # Setup the FSQ Quantizer
            self.use_fsq = self.cfg.get("fsq_quantizer_levels", None) is not None

            if self.use_fsq:
                bottleneck_dim = len(self.cfg.fsq_quantizer_levels)
                hidden_size = self.tts_model.hidden_size
                self.quantizer_bottleneck = nn.Linear(pre_quant_hidden, bottleneck_dim)
                if not self.cfg.get("skip_fsq", False):
                    if self.cfg.get("use_ifsq", False):
                        self.vector_quantizer = ImprovedFiniteScalarQuantizer(self.cfg.fsq_quantizer_levels)
                    else:
                        self.vector_quantizer = FiniteScalarQuantizer(self.cfg.fsq_quantizer_levels)
                self.quantizer_projection = nn.Linear(bottleneck_dim, hidden_size)

        self.frame_length = cfg["data"]["frame_length"]

    def prepare_inputs(self, batch: dict):
        delay_frames = self.cfg.get("num_delay_speech_tokens", 0)
        if delay_frames > 0:
            # Get the exact prompt end in TOKENS/FRAMES (Assuming uniform prompt length in batch)
            prompt_end_frame = batch["non_prompt_mask"][0].float().argmax().item()

            # Convert token boundary to TARGET AUDIO SAMPLES
            samples_per_frame_out = int(self.target_sample_rate * self.frame_length)
            prompt_end_sample_out = int(prompt_end_frame * samples_per_frame_out)
            delay_samples_out = int(delay_frames * samples_per_frame_out)
            
            zeros_pad_out = torch.zeros(
                batch["target_audio"].size(0),
                delay_samples_out,
                device=batch["target_audio"].device,
                dtype=batch["target_audio"].dtype,
            )
            
            # Pad the audio exactly at the prompt boundary
            batch["target_audio"] = torch.cat([
                batch["target_audio"][:, :prompt_end_sample_out], 
                zeros_pad_out, 
                batch["target_audio"][:, prompt_end_sample_out:]
            ], dim=1)
            
            # Increase the audio lengths
            batch["target_audio_lens"] = batch["target_audio_lens"] + delay_samples_out

            # Pad non_prompt_mask
            mask_delay_pad = torch.ones(
                batch["non_prompt_mask"].size(0), 
                delay_frames, 
                device=self.device, 
                dtype=batch["non_prompt_mask"].dtype
            )
            
            batch["non_prompt_mask"] = torch.cat([
                batch["non_prompt_mask"][:, :prompt_end_frame], 
                mask_delay_pad, 
                batch["non_prompt_mask"][:, prompt_end_frame:]
            ], dim=1)

        # The parent will now extract TTS codes where index 0 is pure silence if delay_frames > 0, 
        # preserving your real audio later in the sequence.
        inputs = super().prepare_inputs(batch)

        # -------------------------------------------------------------------
        # 2. Resample for the ASR Encoder
        # -------------------------------------------------------------------
        target_audio_asr_sr = resample(
            batch["target_audio"], self.target_sample_rate, self.cfg.get("asr_sample_rate", 16000)
        )
        
        if self.training:
            target_audio_lens_asr_sr = (
                batch["target_audio_lens"] / self.target_sample_rate * self.cfg.get("asr_sample_rate", 16000)
            ).to(torch.long)
        else:
            # During evaluation, treat the entire padded audio as a valid sequence.
            target_audio_lens_asr_sr = torch.full(
                (target_audio_asr_sr.shape[0],),
                target_audio_asr_sr.shape[1],
                dtype=torch.long,
                device=target_audio_asr_sr.device,
            )

        # -------------------------------------------------------------------
        # 3. THE CUT: Shift ASR audio backward (into the future) and wipe prompt
        # -------------------------------------------------------------------
        if delay_frames > 0:
            # Safely get the integer index of the prompt boundary
            prompt_end_frame = batch["non_prompt_mask"][0].float().argmax().item()
            
            samples_per_frame_asr = int(self.cfg.get("asr_sample_rate", 16000) * self.frame_length)
            delay_samples_asr = int(delay_frames * samples_per_frame_asr)
            prompt_end_sample_asr = int(prompt_end_frame * samples_per_frame_asr)

            # Check if audio is long enough to contain Prompt + Delay
            if target_audio_asr_sr.shape[1] > (prompt_end_sample_asr + delay_samples_asr):
                
                # Identify where the delay ends and the real audio begins
                delay_end_sample_asr = prompt_end_sample_asr + delay_samples_asr

                # Cut out the delay from the middle!
                prompt_part = target_audio_asr_sr[:, :prompt_end_sample_asr]
                real_audio_part = target_audio_asr_sr[:, delay_end_sample_asr:]

                # Pad the END to maintain tensor shape
                zeros_pad_end = torch.zeros(
                    target_audio_asr_sr.size(0),
                    delay_samples_asr,
                    device=target_audio_asr_sr.device,
                    dtype=target_audio_asr_sr.dtype,
                )
                # Zero out the prompt portion safely (In-place operation)
                prompt_part[:, :] = 0.0
                # Re-assemble: Prompt + Real Audio (shifted left) + Pad
                target_audio_asr_sr = torch.cat([prompt_part, real_audio_part, zeros_pad_end], dim=1)
            
                # Correct the sequence lengths
                target_audio_lens_asr_sr = torch.clamp(target_audio_lens_asr_sr - delay_samples_asr, min=1)

        # Generate the ASR embedding
        encoded, encoded_len = self.perception(
            input_signal=target_audio_asr_sr, input_signal_length=target_audio_lens_asr_sr
        )

        if self.cfg.get("use_pretrained_quantizer", False):
            encoded = self.quantizer_projection(encoded)
        else:
            # Apply FSQ Quantization
            if self.use_fsq:
                z = self.quantizer_bottleneck(encoded)
                with fp32_precision():
                    if not self.cfg.get("skip_fsq", False):
                        z_q, _ = self.vector_quantizer(inputs=z.transpose(1, 2), input_len=encoded_len)
                    else:
                        z_q = z.transpose(1, 2)

                z_q = z_q.transpose(1, 2).to(z.dtype)
                encoded = self.quantizer_projection(z_q)

        # Align sequence lengths to the target codes
        target_len = inputs["code"].shape[1]
        if encoded.shape[1] < target_len:
            encoded = F.pad(encoded, (0, 0, 0, target_len - encoded.shape[1]))
        elif encoded.shape[1] > target_len:
            encoded = encoded[:, :target_len, :]

        inputs["asr_speech_tokens_emb"] = encoded

        # Wipe text inputs
        inputs["subword_ids"] = None
        inputs["subword_mask"] = None
        inputs["context_hidden_state"] = None

        return inputs

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.tts_model,):
            if is_frozen(m):
                m.eval()

        inputs = self.prepare_inputs(batch)

        tts_output = self.tts_model(
            code=inputs["code"],
            audio_mask=inputs["audio_mask"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            context_hidden_state=None,
            subword_ids=None,
            subword_mask=None,
            non_prompt_mask=inputs["non_prompt_mask"],
            dataset_type=batch.get("dataset_type", None),
            tiled_prompt_audio_codes=inputs["tiled_prompt_audio_codes"],
            tiled_prompt_subword_ids=inputs["tiled_prompt_subword_ids"],
            tiled_prompt_subword_mask=inputs["tiled_prompt_subword_mask"],
            asr_speech_tokens_emb=inputs["asr_speech_tokens_emb"],
        )

        loss_dict = {"lm_loss": tts_output.lm_loss, "c_loss": tts_output.c_loss, "k_loss": tts_output.k_loss}
        loss = sum(loss_dict.values())
        num_frames = inputs["output_lens"].sum()
        B, T = inputs["code"].shape[:2]

        ans = {
            "loss": loss,
            "learning_rate": torch.as_tensor(
                self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0
            ),
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),
            "padding_ratio": num_frames / (B * T),
            **loss_dict,
        }
        self.log_dict(ans, on_step=True)
        return ans

    def get_teacher_force_inference_audio(self, batch, guidance_enabled=True):
        inputs = self.prepare_inputs(batch)

        tts_output = self.tts_model(
            code=inputs["code"],
            audio_mask=inputs["audio_mask"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            context_hidden_state=None,
            subword_ids=None,
            subword_mask=None,
            non_prompt_mask=inputs["non_prompt_mask"],
            generation_config=self._get_generation_config(guidance_enabled=guidance_enabled),
            teacher_forcing_inference=True,
            guidance_enabled=guidance_enabled,
            asr_speech_tokens_emb=inputs["asr_speech_tokens_emb"],
        )
        tf_audio_codes_pred = tts_output["codes"].squeeze(2)

        tf_audio_codes_pred = replace_control_speech_codes(
            tf_audio_codes_pred, self._control_codes, self.codec_silence_tokens
        )
        with ensures_target_precision(self.audio_codec_run_dtype), torch.no_grad():
            audio_pred, audio_len = self.audio_codec.decode(tf_audio_codes_pred, inputs["output_lens"])

        return audio_pred.squeeze(1), audio_len

    @torch.no_grad()
    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            B = len(dataset_batch['sample_id'])

            # run inference for a custom speaker reference
            if self.cfg.get("inference_speaker_reference", None):
                new_dataset_batch = copy.deepcopy(dataset_batch)
                speaker_audio, sr = load_audio_librosa(self.cfg.inference_speaker_reference)
                speaker_audio = resample(speaker_audio, sr, self.target_sample_rate)
                speaker_audio = speaker_audio.repeat(B, 1).to(self.device)
                # lengths -> [B]
                speaker_audio_lens = torch.tensor([speaker_audio.size(1)], device=self.device).long().repeat(B)
                new_dataset_batch["audio_prompt"] = speaker_audio
                new_dataset_batch["audio_prompt_lens"] = speaker_audio_lens
                self.run_evaluation_one_batch(name, new_dataset_batch, use_dataloader_init=False)

            # run inference using dataloader speaker references
            else:
                self.run_evaluation_one_batch(name, dataset_batch, use_dataloader_init=False)

    @torch.no_grad()
    def infer_codes_one_step(
        self,
        current_asr_emb,
        current_subword_mask,
        prev_audio_tokens,
        past_key_values,
        guidance_enabled=True,
        generation_config=None,
        ignore_eos_flag_stop=True,
        tiled_prompt_audio_codes=None,
        tiled_prompt_subword_ids=None,
        tiled_prompt_subword_mask=None,
    ):
        inputs = {
            "code": prev_audio_tokens,
            "context_hidden_state": None,
            "subword_ids": None,
            "subword_mask": None,
            "past_key_values": past_key_values,
            "use_cache": True,
            "guidance_enabled": guidance_enabled,
            "generation_config": generation_config,
            "ignore_eos_flag_stop": ignore_eos_flag_stop,
            "tiled_prompt_audio_codes": tiled_prompt_audio_codes,
            "tiled_prompt_subword_ids": tiled_prompt_subword_ids,
            "tiled_prompt_subword_mask": tiled_prompt_subword_mask,
            "asr_speech_tokens_emb": current_asr_emb,
        }

        outputs = self.tts_model(**inputs)
        return outputs["codes"], outputs["past_key_values"]

    @torch.no_grad()
    def decode_one_audio_step(self, gen_audio_codes_history, number_prev_tokens=None):
        """
        Decodes one step of generated audio codec tokens to raw waveform.

        Args:
            gen_audio_codes_history (torch.Tensor): Audio tokens history, shape (B, T, C).
            number_prev_tokens (int, optional): Number of previous tokens to decode, for incremental decoding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - audio_pred_cur_step: Latest decoded waveform chunk, shape (B, wav_to_token_ratio).
                - audio_len: Lengths (number of samples), shape (B,).
        """
        with fp32_precision(), torch.no_grad():
            if number_prev_tokens:
                gen_audio_codes_history = gen_audio_codes_history[:, -number_prev_tokens:]

            gen_audio_codes_history = replace_control_speech_codes(
                gen_audio_codes_history, self._control_codes, self.codec_silence_tokens
            )
            gen_audio_codes_lens = torch.tensor(
                [gen_audio_codes_history.size(1)] * gen_audio_codes_history.size(0), device=self.device
            )
            audio_pred, audio_len = self.audio_codec.decode(gen_audio_codes_history, gen_audio_codes_lens)

        # return only the current/lastest audio chunk
        audio_pred_cur_step = audio_pred.squeeze(1)[:, -self.audio_codec.config.wav_to_token_ratio :]
        audio_len[:] = self.audio_codec.config.wav_to_token_ratio
        return audio_pred_cur_step, audio_len

    @torch.no_grad()
    def offline_inference(
        self,
        next_asr_embs: torch.Tensor,
        init_inputs: dict,
        task: str = "",
        guidance_enabled: bool = True,
        generation_config: dict = None,
        incremental_audio_decoding: bool = False,
    ) -> dict[str, torch.Tensor]:

        B = next_asr_embs.size(0)

        if self.cfg.tts_config.get("use_tiled_prompt_channel", False):
            base_prompt_audio_codes = init_inputs.pop("base_prompt_audio_codes")
            base_prompt_subword_ids = init_inputs.pop("base_prompt_subword_ids")
            p_lens = init_inputs.pop("p_lens")
            safe_p_lens = p_lens.clamp_min(1)

        if generation_config is None:
            generation_config = self._get_generation_config(guidance_enabled)
        
        print("Doing inference with the following generation_config:", generation_config, "CFG enabled?", guidance_enabled)

        init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": guidance_enabled})

        # Warmup the model with prompt inputs
        outputs = self.tts_model(**init_inputs)

        if self.cfg.get("inference_skip_first_code_prediction_on_init", True):
            code = init_inputs["code"][:, -1:]
        else:
            code, _, _ = self.tts_model.generate_step(outputs.hidden_states[:, -1:], **generation_config)

        past_key_values = outputs["past_key_values"]
        max_steps = next_asr_embs.size(1)

        gen_audio_codes = torch.zeros(
            B, max_steps, self.tts_model.config.num_quantizers, device=self.device, dtype=torch.long
        )

        audio_pred = None
        audio_pred_len = torch.zeros(B, device=self.device, dtype=torch.long)

        for i in range(max_steps):
            # Extract current step ASR embedding [B, 1, H]
            current_asr_emb = next_asr_embs[:, i].unsqueeze(1)

            if self.cfg.tts_config.get("use_tiled_prompt_channel", False):
                t_abs = p_lens + i
                mod_idx = (t_abs % safe_p_lens).unsqueeze(1)
                step_tiled_text = torch.gather(base_prompt_subword_ids, 1, mod_idx)
                C = base_prompt_audio_codes.shape[-1]
                gather_indices_audio = mod_idx.unsqueeze(-1).expand(-1, -1, C)
                step_tiled_audio = torch.gather(base_prompt_audio_codes, 1, gather_indices_audio)
                step_tiled_mask = torch.ones_like(step_tiled_text, dtype=torch.bool)
            else:
                step_tiled_audio = None
                step_tiled_text = None
                step_tiled_mask = None

            current_subword_mask = torch.ones(B, 1, device=self.device, dtype=torch.bool)

            code, past_key_values = self.infer_codes_one_step(
                current_asr_emb=current_asr_emb,
                current_subword_mask=current_subword_mask,
                prev_audio_tokens=code,
                past_key_values=past_key_values,
                guidance_enabled=guidance_enabled,
                generation_config=generation_config,
                ignore_eos_flag_stop=True,
                tiled_prompt_audio_codes=step_tiled_audio,
                tiled_prompt_subword_ids=step_tiled_text,
                tiled_prompt_subword_mask=step_tiled_mask,
            )

            # --- THE AR SILENCE BYPASS ---
            # Forcibly overwrite the sampler's predictions with absolute silence 
            # during the algorithmic delay window to prevent hallucination cascades.
            delay_frames = self.cfg.get("num_delay_speech_tokens", 0)
            if i < delay_frames and self.cfg.get("force_silence_on_delay", False):
                for q in range(code.size(2)):
                    code[:, :, q] = self.codec_silence_tokens[q]

            gen_audio_codes[:, i] = code.squeeze(1)

            if incremental_audio_decoding:
                audio_pred_i, audio_pred_i_len = self.decode_one_audio_step(
                    gen_audio_codes[:, : i + 1],
                    number_prev_tokens=self.cfg.get("inference_codec_decoding_prev_tokens_number", None),
                )
                if audio_pred is None:
                    audio_pred = audio_pred_i
                else:
                    audio_pred = torch.cat([audio_pred, audio_pred_i], dim=1)
                audio_pred_len += audio_pred_i_len

        if not incremental_audio_decoding:
            gen_audio_codes_lens = torch.tensor([gen_audio_codes.shape[1]] * gen_audio_codes.shape[0]).to(self.device)
            gen_audio_codes = replace_control_speech_codes(
                gen_audio_codes, self._control_codes, self.codec_silence_tokens
            )
            with ensures_target_precision(self.audio_codec_run_dtype), torch.no_grad():
                audio_pred, audio_pred_len = self.audio_codec.decode(gen_audio_codes, gen_audio_codes_lens)

        return audio_pred.squeeze(1), audio_pred_len

    def set_init_inputs(self, speaker_audio=None, speaker_audio_lens=None, system_prompt=None, speaker_name=None):
        # 1. Let the parent handle the complex setup, codec encoding, and caching of base tensors
        init_inputs = super().set_init_inputs(speaker_audio, speaker_audio_lens, system_prompt, speaker_name)

        # 2. Recreate the exact target_audio that the parent just built so we can pass it to the ASR encoder
        with fp32_precision():
            prompt_audio_size = int(
                ((self.data_cfg.audio_prompt_duration * self.target_sample_rate) // self.target_samples_per_frame)
                * self.target_samples_per_frame
            )

            if speaker_name is not None:
                speaker_audio = torch.zeros((1, prompt_audio_size), device=self.device, dtype=torch.float32)
                speaker_audio_lens = torch.LongTensor([speaker_audio.shape[1]]).to(self.device)

            B, T = speaker_audio.shape
            prompt_audio = torch.zeros(B, prompt_audio_size, device=self.device, dtype=speaker_audio.dtype)

            for b in range(B):
                valid_len = min(speaker_audio_lens[b].item(), T)
                if valid_len <= 0:
                    continue
                valid_segment = speaker_audio[b, :valid_len]
                if valid_len >= prompt_audio_size:
                    prompt_audio[b] = valid_segment[:prompt_audio_size]
                else:
                    repeat_factor = (prompt_audio_size + valid_len - 1) // valid_len
                    expanded = valid_segment.repeat(repeat_factor)
                    prompt_audio[b] = expanded[:prompt_audio_size]

            prompt_audio[:, -int(self.target_samples_per_frame * 2) :] = 0

            # Recreate the text pad sizing
            if system_prompt is not None and self.cfg.get("use_system_prompt", None) and system_prompt != "":
                text_prompt = torch.as_tensor(
                    [self.tokenizer.bos] + self.tokenizer.text_to_ids(system_prompt) + [self.tokenizer.eos],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                text_prompt = torch.tensor([self.tokenizer.eos], dtype=torch.long, device=self.device)

            pad_size = text_prompt.size(-1) * self.target_samples_per_frame
            pad_audio = (
                torch.zeros(pad_size, device=prompt_audio.device, dtype=prompt_audio.dtype).unsqueeze(0).repeat(B, 1)
            )

            target_audio = torch.cat([pad_audio, prompt_audio], dim=1)
            target_audio_len = torch.tensor([target_audio.size(-1)] * B, dtype=torch.long, device=self.device)

        # 3. Get ASR embeddings for this recreated target_audio
        asr_sr = self.cfg.get("asr_sample_rate", 16000)
        target_audio_asr_sr = resample(target_audio, self.target_sample_rate, asr_sr)
        target_audio_lens_asr_sr = (target_audio_len / self.target_sample_rate * asr_sr).to(torch.long)

        # Apply the exact same lookahead shift we designed earlier!
        # delay_frames = self.cfg.get("num_delay_speech_tokens", 0)
        # if delay_frames > 0:
        #     samples_per_frame_asr = int(self.cfg.get("asr_sample_rate", 16000) * self.frame_length)
        #     delay_samples_asr = int(delay_frames * samples_per_frame_asr)
        #     if target_audio_asr_sr.shape[1] > delay_samples_asr:
        #         shifted_audio = target_audio_asr_sr[:, delay_samples_asr:]
        #         zeros_pad = torch.zeros(
        #             target_audio_asr_sr.size(0),
        #             delay_samples_asr,
        #             device=target_audio_asr_sr.device,
        #             dtype=target_audio_asr_sr.dtype,
        #         )
        #         target_audio_asr_sr = torch.cat([shifted_audio, zeros_pad], dim=1)
        target_audio_asr_sr = target_audio_asr_sr * 0.0
        encoded, encoded_len = self.perception(
            input_signal=target_audio_asr_sr, input_signal_length=target_audio_lens_asr_sr
        )
        if self.cfg.get("use_pretrained_quantizer", False):
            encoded = self.quantizer_projection(encoded)
        else:
            if self.use_fsq:
                z = self.quantizer_bottleneck(encoded)
                with fp32_precision():
                    if not self.cfg.get("skip_fsq", False):
                        z_q, _ = self.vector_quantizer(inputs=z.transpose(1, 2), input_len=encoded_len)
                    else:
                        z_q = z.transpose(1, 2)
                z_q = z_q.transpose(1, 2).to(z.dtype)
                encoded = self.quantizer_projection(z_q)

        # The parent init_inputs drops the last frame (`[:, :-1]`), so target_len is code length + 1
        target_len = init_inputs["code"].shape[1] + 1
        if encoded.shape[1] < target_len:
            encoded = F.pad(encoded, (0, 0, 0, target_len - encoded.shape[1]))
        elif encoded.shape[1] > target_len:
            encoded = encoded[:, :target_len, :]

        # 4. Inject into init_inputs (applying the same [:, :-1] slice)
        init_inputs["asr_speech_tokens_emb"] = encoded[:, :-1]

        # 5. Nullify text inputs securely
        init_inputs["subword_ids"] = None
        init_inputs["subword_mask"] = None
        init_inputs["context_hidden_state"] = None

        # 6. Update the cache so get_init_inputs finds them later
        self._init_input_cache["asr_speech_tokens_emb"] = init_inputs["asr_speech_tokens_emb"].detach().clone()
        self._init_input_cache["subword_ids"] = None
        self._init_input_cache["subword_mask"] = None
        self._init_input_cache["context_hidden_state"] = None

        return init_inputs

    def get_init_inputs(self, B: int, init_inputs_names=None):
        if init_inputs_names is None:
            init_inputs_names = [
                "code",
                "audio_mask",
                "non_prompt_mask",
            ]

        # Ensure our custom embedding is requested from the cache
        if "asr_speech_tokens_emb" not in init_inputs_names:
            init_inputs_names.append("asr_speech_tokens_emb")

        init_inputs = super().get_init_inputs(B, init_inputs_names)

        # Ensure textual inputs are explicitly None so the generative codec relies strictly on ASR
        init_inputs["subword_ids"] = None
        init_inputs["subword_mask"] = None
        init_inputs["context_hidden_state"] = None

        return init_inputs

    def on_train_epoch_start(self) -> None:
        # Call the parent's method (which ensures codec precision)
        super().on_train_epoch_start()

        # This prevents inference tensors from leaking into the training graph
        if hasattr(self, "_init_input_cache"):
            self._init_input_cache.clear()

    @torch.no_grad()
    def run_evaluation_one_batch(self, name, dataset_batch, use_dataloader_init=False):
        results = {}
        inputs = self.prepare_inputs(dataset_batch)
        asr_emb = inputs["asr_speech_tokens_emb"]

        results["audio_tf"], results["audio_tf_len"] = self.get_teacher_force_inference_audio(dataset_batch)

        if use_dataloader_init:
            init_inputs = {
                "code": inputs["code"],
                "audio_mask": inputs["audio_mask"],
                "non_prompt_mask": inputs["non_prompt_mask"],
                "asr_speech_tokens_emb": asr_emb,
            }
            for key in init_inputs:
                if init_inputs[key] is not None:
                    init_inputs[key] = torch.stack(
                        [init_inputs[key][i, :plen] for i, plen in enumerate(dataset_batch["prompt_lens"])]
                    )
        else:
            sp = dataset_batch.get("system_prompts_raw")
            system_prompt = sp[0] if sp else None
            self.set_init_inputs(
                speaker_audio=dataset_batch["audio_prompt"],
                speaker_audio_lens=dataset_batch["audio_prompt_lens"],
                system_prompt=system_prompt,
            )
            init_inputs = self.get_init_inputs(B=inputs["code"].size(0))

            # Inject corresponding prompt portion of the ASR embedding into init_inputs
            target_len = init_inputs["code"].shape[1]
            init_inputs["asr_speech_tokens_emb"] = torch.stack(
                [asr_emb[i, :target_len] for i in range(asr_emb.size(0))]
            )

        # Slice remaining ASR embeddings to represent the 'next' sequences to generate
        next_asr_embs = torch.stack([asr_emb[i, plen:] for i, plen in enumerate(dataset_batch["prompt_lens"])])

        results["audio"], results["audio_len"] = self.offline_inference(
            next_asr_embs=next_asr_embs,
            init_inputs=init_inputs,
            task=dataset_batch["task"][0],
        )

        dataset_batch["source_audio"] = dataset_batch["source_audio"][
            :, -int(next_asr_embs.size(1) * self.source_samples_per_frame) :
        ]

        results["audio_tf"] = results["audio_tf"][:, -int(next_asr_embs.size(1) * self.target_samples_per_frame) :]

        target_audio_no_prompt = dataset_batch["target_audio"][
            :, -int(next_asr_embs.size(1) * self.target_samples_per_frame) :
        ]
        target_audio_no_prompt_lens = dataset_batch["target_audio_lens"] - (
            torch.tensor(
                dataset_batch["prompt_lens"], dtype=torch.long, device=dataset_batch["target_audio_lens"].device
            )
            * self.target_samples_per_frame
        )

        results["audio_len"] = target_audio_no_prompt_lens.clone()
        delay_frames = self.cfg.get("num_delay_speech_tokens", 0) + 4 # give more 0.32s to avoid cuts given it is a generative model
        if delay_frames:
            samples_per_frame_out = int(self.target_sample_rate * self.frame_length)
            delay_samples_out = int(delay_frames * samples_per_frame_out)
            results["audio_len"] = results["audio_len"] + delay_samples_out
        
        # Cap the recon_lens so it doesn't exceed the actual generated tensor size
        max_generated_samples = results["audio"].shape[1]
        results["audio_len"] = torch.clamp(results["audio_len"], max=max_generated_samples)

        with fp32_precision():
            metric_audio_pred = results["audio"]
            metric_audio_pred_lens = results["audio_len"]

            metric_audio_pred = resample(metric_audio_pred, self.target_sample_rate, 16000)
            metric_audio_pred_lens = (metric_audio_pred_lens / self.target_sample_rate * 16000).to(torch.long)
            target_audio_no_prompt_16khz = resample(target_audio_no_prompt, self.target_sample_rate, 16000)
            target_audio_no_prompt_lens_16khz = (target_audio_no_prompt_lens / self.target_sample_rate * 16000).to(
                torch.long
            )

            if self.cfg.get("use_GT_transcriptions_for_metrics", True):
                target_asr_texts = self.asr_bleu.asr.transcribe(
                    [
                        audio[:alen]
                        for audio, alen in zip(target_audio_no_prompt_16khz, target_audio_no_prompt_lens_16khz)
                    ],
                    batch_size=target_audio_no_prompt_16khz.shape[0],
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

            self.intelligibility.update(
                name=name + "_gt",
                refs=dataset_batch["target_texts"],
                pred_audio=target_audio_no_prompt_16khz,
                pred_audio_lens=target_audio_no_prompt_lens_16khz,
                asr_hyps=(metric_text if self.cfg.get("use_GT_transcriptions_for_metrics", True) else None),
            )

            self.secs.update(
                name=name,
                target_audio=resample(dataset_batch["target_audio"], self.target_sample_rate, 16000),
                target_audio_lens=(dataset_batch["target_audio_lens"] / self.target_sample_rate * 16000).to(
                    torch.long
                ),
                pred_audio=resample(results["audio"], self.target_sample_rate, 16000),
                pred_audio_lens=(results["audio_len"] / self.target_sample_rate * 16000).to(torch.long),
            )

            # NOTE: EOU labels are skipped since we dropped the textual representation entirely
            # eou_labels = ...

            self.results_logger.update(
                name=name,
                refs=dataset_batch["target_texts"],
                hyps=metric_text,
                asr_hyps=asr_hyps,
                samples_id=dataset_batch['sample_id'],
                pred_audio=results["audio"].float(),
                pred_audio_tf=results["audio_tf"].float(),
                pre_audio_trimmed=None,
                reference_audio=dataset_batch["audio_prompt"].float(),
                target_audio=target_audio_no_prompt.float(),
                pred_audio_sr=self.target_sample_rate,
                user_audio=dataset_batch["source_audio"].float(),
                user_audio_sr=self.source_sample_rate,
                eou_pred=None,
                fps=self.target_fps,
                results=results if self.cfg.get("dump_tokens_text", False) else None,
                tokenizer=self.tokenizer,
            )
