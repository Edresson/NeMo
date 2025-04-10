import numpy as np
import math
import torch

from transformers import MimiConfig
from transformers.models.mimi.modeling_mimi import MimiEncoder, MimiTransformerModel, MimiConv1d, MimiConvTranspose1d, MimiDecoder
from nemo.core.classes.module import NeuralModule


class MimiAudioEncoder(NeuralModule):
    def __init__(self, out_size=32, sampling_rate=24000, upsampling_ratios=[8, 6, 5, 4], frame_rate=12.5, is_causal=True, hidden_size=512):
        super().__init__()
        # get Mimi default config
        self.config = MimiConfig()

        # redefine configs based on nemo configs
        self.config.frame_rate = frame_rate
        self.config.sampling_rate = sampling_rate
        self.config.upsampling_ratios = upsampling_ratios
        self.config.use_causal_conv = is_causal
        self.config.hidden_size = hidden_size

        # define upsampling rate
        self.downsampling_rate = self.config.sampling_rate / self.config.frame_rate

        self.encoder = MimiEncoder(self.config)
        self.encoder_transformer = MimiTransformerModel(self.config)

        # extra downsample requeried because MiMiEncoder works in a different frame rate
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
        embeddings = self.downsample(embeddings)
        embeddings = self.out_projection(embeddings)

        # compute output_len based on downsampling rate
        output_len = (audio_len / self.downsampling_rate).long()
        return embeddings, output_len


class MimiAudioDecoder(NeuralModule):
    def __init__(self, input_size=32, sampling_rate=24000, upsampling_ratios=[8, 6, 5, 4], frame_rate=12.5, is_causal=True, hidden_size=512):
        super().__init__()
        # get Mimi default config
        self.config = MimiConfig()

        # redefine configs based on nemo configs
        self.config.frame_rate = frame_rate
        self.config.sampling_rate = sampling_rate
        self.config.upsampling_ratios = upsampling_ratios
        self.config.use_causal_conv = is_causal
        self.config.hidden_size = hidden_size
        # define upsampling rate
        self.upsampling_rate = self.config.sampling_rate / self.config.frame_rate

        self.decoder_transformer = MimiTransformerModel(self.config)
        self.decoder = MimiDecoder(self.config)

        # extra upsampling requeried because MiMiEncoder works in a different frame rate
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