import math
from typing import List
from typing import Union

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn

from sa_edm.dac.model.base import CodecMixin
from sa_edm.dac.nn.layers import Snake1d
from sa_edm.dac.nn.layers import WNConv1d
from sa_edm.dac.nn.layers import WNConvTranspose1d
from sa_edm.dac.model.distributions import DiagonalGaussianDistribution
from torch.nn.utils import remove_weight_norm, weight_norm


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
        double_z: bool = False, 
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)] # TODO; if we wanna do stereo case, 1 will be 2.

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        if double_z:
            self.block += [
                Snake1d(d_model),
                WNConv1d(d_model, 2*d_latent, kernel_size=3, padding=1), 
            ]            

        else:
            self.block += [
                Snake1d(d_model),
                WNConv1d(d_model, d_latent, kernel_size=3, padding=1), 
            ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class GaussianDAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = 8,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        sample_rate: int = 44100,
        double_z: bool = True,

    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim, double_z)

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        
        self.pre_conv = WNConv1d(2*latent_dim, 2*latent_dim, 1) 
        self.post_conv = WNConv1d(latent_dim, latent_dim, 1)
        
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
    ):
        """Encode given audio data and return latent features will be sampled from Gaussian distribution.

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode

        Returns
        -------
        posterir distribution
        
        """
        z = self.encoder(audio_data)
        moments = self.pre_conv(z)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z: torch.Tensor):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input

        Returns
        -------
        Reconstructed audio : Tensor[B x 1 x length]

        """
        z = self.post_conv(z)
        recon = self.decoder(z)
        return recon
        # return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        sample_posterior: bool = True,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Continuous latent representation of input
            "kl_loss" : Tensor[1]
                KL loss
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        
        posterior = self.encode(audio_data)
        
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        
        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "kl_loss": kl_loss,
        }

    def encode_to_latent(
        self, 
        audio_data: torch.Tensor, 
        sample_rate: int = None, 
        sample_posterior: bool = True,
    ):
        """Encode given audio data and return latent features

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode

        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`

        Returns
        -------
        "z" : Tensor[B x D x T]
            Continuous representation of input
        """
        audio_data = self.preprocess(audio_data, sample_rate)
        posterior = self.encode(audio_data)
        
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        return z

    def decode_to_waveform(self, z: torch.Tensor):
        """Decode given continous latent and return audio datas

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        return self.decode(z)

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for module in self.modules():
            if isinstance(module, weight_norm.WeightNorm):
                remove_weight_norm(module)


class AEDAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = 8,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        sample_rate: int = 44100,
        latent_noise: float=0.,
        # double_z: bool = True,

    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate
        self.latent_noise = max(latent_noise, 0)

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        
        self.pre_conv = WNConv1d(latent_dim, latent_dim, 1) 
        self.post_conv = WNConv1d(latent_dim, latent_dim, 1)
        
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        stochastic_latent=True
    ):
        """Encode given audio data and return latent features will be sampled from Gaussian distribution.

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode

        Returns
        -------
        latent represenataion
        
        """
        z = self.encoder(audio_data)
        h = self.pre_conv(z)
        if stochastic_latent:
            h = h + self.latent_noise * torch.randn_like(h)
        return h

    def decode(self, z: torch.Tensor):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input

        Returns
        -------
        Reconstructed audio : Tensor[B x 1 x length]

        """
        z = self.post_conv(z)
        recon = self.decoder(z)
        return recon
        # return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        stochastic_latent=True,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Continuous latent representation of input
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        
        z = self.encode(audio_data, stochastic_latent)
        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
        }

    def get_latent(
        self, 
        audio_data: torch.Tensor, 
        sample_rate: int = None, 
        stochastic_latent: bool = True,
    ):
        """Encode given audio data and return latent features

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode

        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`

        Returns
        -------
        "z" : Tensor[B x D x T]
            Continuous representation of input
        """
        audio_data = self.preprocess(audio_data, sample_rate)
        z = self.encode(audio_data, stochastic_latent)
        
        return z

    def recon_from_clatent(self, z: torch.Tensor):
        """Decode given continous latent and return audio datas
        TODO; Perhaps need to freeze weights when integrate to LDM

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        return self.decode(z)

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for module in self.modules():
            if isinstance(module, weight_norm.WeightNorm):
                remove_weight_norm(module)
