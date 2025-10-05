import torch
import torch.nn as nn
from nemo.collections.asr.modules import ConformerEncoder
from nemo.core.classes.common import typecheck
from nemo.collections.tts.modules.audio_codec_modules import FiniteScalarQuantizer

from contextlib import contextmanager
@contextmanager
def fp32_precision():
    """
    Workaround for precision related issues when training with bf16-true PyTorch Lightning precision setting.
    In bf16-true, PTL changes PyTorch's default dtype, which may break implicit assumptions for some models.
    This context manager restores default float32 precision and runs the computation in float32 autocast context.
    """
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32):
            yield
    finally:
        torch.set_default_dtype(default_dtype)


class ConformerEncoderWithFSQ(ConformerEncoder):
    """
    A ConformerEncoder extension that adds a Finite Scalar Quantizer (FSQ)
    module at the output. Behavior is identical to ConformerEncoder by default.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            quantizer_levels (List[int], optional):
                List of quantization levels per bottleneck dimension, e.g. [8, 8, 8, 8].
                If provided, FSQ layers will be added after the encoder output.

        Example:
            encoder = ConformerEncoderWithFSQ(
                feat_in=80,
                n_layers=12,
                d_model=512,
                feat_out=256,
                quantizer_levels=[8, 7, 6, 6],
            )
        """
        quantizer_levels = kwargs.pop("quantizer_levels", None)
        super().__init__(*args, **kwargs)

        self.use_fsq = quantizer_levels is not None
        if self.use_fsq:
            bottleneck_dim = len(quantizer_levels)
            # out_dim = self._feat_out  # use feat_out if provided, else d_model
            out_dim = self._feat_out if self._feat_out != -1 else self.d_model

            # Bottleneck projection → Quantizer → Projection back
            self.quantizer_bottleneck = nn.Linear(out_dim, bottleneck_dim)
            self.vector_quantizer = FiniteScalarQuantizer(quantizer_levels)
            self.quantizer_projection = nn.Linear(bottleneck_dim, out_dim)

    @typecheck()
    def forward(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        bypass_pre_encode=False,
    ):
        """
        Forward pass — identical to base ConformerEncoder, but applies FSQ at the end if configured.
        """
        outputs = super().forward(
            audio_signal=audio_signal,
            length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            bypass_pre_encode=bypass_pre_encode,
        )

        # Base encoder returns (encoded, encoded_lengths, *optional_caches)
        if not self.use_fsq:
            return outputs

        encoded, encoded_len = outputs[:2]
        other = outputs[2:]
    
        # Apply quantization: Linear → FSQ → Linear
        z = self.quantizer_bottleneck(encoded.transpose(1, 2))
        dtype = z.dtype
        # keep finite scalar quantization on FP32
        with fp32_precision():
            # add tanh to avoid overflow
            z = torch.tanh(z)
            z_q, _ = self.vector_quantizer(inputs=z.transpose(1, 2), input_len=None)
        # make sure that it is runing in the right dtype
        z_q = z_q.transpose(1, 2).to(dtype)
        encoded_quantized = self.quantizer_projection(z_q).transpose(1, 2)

        return (encoded_quantized, encoded_len, *other)
