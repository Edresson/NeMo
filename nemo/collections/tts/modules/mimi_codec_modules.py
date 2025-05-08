import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import MimiConfig
from transformers.models.mimi.modeling_mimi import MimiEncoder, MimiTransformerModel, MimiConv1d, MimiConvTranspose1d, MimiDecoder
from nemo.core.classes.module import NeuralModule

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
    ):
        super().__init__()


        self.samples_per_frame = samples_per_frame
        self.audio_proj_size = audio_proj_size
        self.output_dim = output_dim

        self.config = MimiConfig()
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

        out = self.inp_projection_no_bias(audio)
        out = self.inp_projection(out)
        out = self.layers(out)[0]
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
    ):
        super().__init__()


        self.samples_per_frame = samples_per_frame
        self.audio_proj_size = audio_proj_size

        self.config = MimiConfig()
        self.config.use_causal_conv = is_causal
        self.config.num_hidden_layers = n_layers
        self.config.intermediate_size = d_ffn
        self.config.hidden_size = d_model
        self.config.sliding_window = sliding_window_size
        self.layers = MimiTransformerModel(self.config)

        self.inp_projection = nn.Linear(input_dim, d_model)
        self.out_projection = nn.Linear(d_model, audio_proj_size)
        self.out_projection_no_bias = nn.Linear(audio_proj_size, samples_per_frame, bias=False)

    def forward(self, inputs, input_len):
        encoded_len = input_len
        out = self.inp_projection(inputs.transpose(1, 2))
        out = self.layers(out)[0]

        out = self.out_projection(out)
        audio = self.out_projection_no_bias(out)

        # resample audio to size
        audio = audio.reshape(inputs.size(0), -1)
        audio_len = (input_len*self.samples_per_frame).int()
        return audio, audio_len


class MimiAudioEncoder(NeuralModule):
    def __init__(self, out_size=32, sampling_rate=24000, upsampling_ratios=[8, 6, 5, 4], frame_rate=12.5, is_causal=True, hidden_size=512, sliding_window=250, num_transformer_layers=8):
        super().__init__()
        # get Mimi default config
        self.config = MimiConfig()

        # redefine configs based on nemo configs
        self.config.frame_rate = frame_rate
        self.config.sampling_rate = sampling_rate
        self.config.upsampling_ratios = upsampling_ratios
        self.config.use_causal_conv = is_causal
        self.config.hidden_size = hidden_size
        self.config.sliding_window = sliding_window
        self.config.num_hidden_layers = num_transformer_layers

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

    def forward(self, audio, audio_len):
        audio = audio.unsqueeze(1)
        embeddings = self.encoder(audio)
        embeddings = self.encoder_transformer(
            embeddings.transpose(1, 2)
        )[0].transpose(1, 2)

        if self.use_extra_downsample:
            embeddings = self.downsample(embeddings)

        embeddings = self.out_projection(embeddings)

        # compute output_len based on downsampling rate
        output_len = (audio_len / self.downsampling_rate).long()
        return embeddings, output_len


class MimiAudioDecoder(NeuralModule):
    def __init__(self, input_size=32, sampling_rate=24000, upsampling_ratios=[8, 6, 5, 4], frame_rate=12.5, is_causal=True, hidden_size=512, sliding_window=250, num_transformer_layers=8):
        super().__init__()
        # get Mimi default config
        self.config = MimiConfig()

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
        embeddings = self.in_projection(inputs)
        if self.use_extra_upsample:
            embeddings = self.upsample(embeddings)

        decoder_outputs = self.decoder_transformer(
            embeddings.transpose(1, 2), past_key_values=past_key_values, return_dict=return_dict
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

# Debug
# mimiencoder = MimiAudioEncoder()
# audio = torch.ones([2, 48000])
# audio_len = torch.zeros(audio.size(0))
# audio_len = audio_len + audio.size(1)
# unquantized_latent, unquantized_latent_len = mimiencoder(audio, audio_len)
# print("unquantized_latent:", unquantized_latent.shape, unquantized_latent_len)
# mimidecoder = MimiAudioDecoder()
# audio_out = mimidecoder(unquantized_latent, unquantized_latent_len)
# print("Audio output", audio_out[0].shape, audio_out[1])
# convert checkpoint
"""
import torch
from transformers import MimiModel
model = MimiModel.from_pretrained("kyutai/mimi")
state_dict = model.state_dict()
for key in list(state_dict.keys()):
    if "encoder." in key or "encoder_transformer." in key or "downsample." in key:
        state_dict["audio_encoder."+key] = state_dict[key]
        del state_dict[key]
    elif "decoder." in key or "decoder_transformer." in key or "upsample." in key:
        state_dict["audio_decoder."+key] = state_dict[key]
        del state_dict[key]
    elif "quantizer." in key:
        del state_dict[key]
    else:
        print("Key not converted!", key)

print(state_dict.keys())
state_dict_new = {'state_dict':state_dict} 
torch.save(state_dict, "/home/ecasanova/Projects/Checkpoints/MimiCodec/mimi_converted_to_nemo.ckpt")
"""