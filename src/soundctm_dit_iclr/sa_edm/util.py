"""
Various utilities for neural networks.
"""

import math
import copy
import torch
import torch as th
import torch.nn as nn
import torchaudio
import random
import itertools
import numpy as np
from torch import nn
from typing import Tuple
import os
import requests
import yaml
from urllib.parse import urljoin
import argparse

def calculate_mean_std(dataset, encoder, device):
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    encoder = encoder.to(device)
    # latent_list = []
    for i, batch in enumerate(loader):
        _, audio, _, _ = batch
        latent = encoder.encode_to_latent(audio.to(device))
        # latent_list.append(latent)
        if i == 0:
            sum_  = torch.sum(latent, dim=(0, 2))
            sum_sq  = torch.sum(latent**2, dim=(0, 2))
            n_samples = latent.size(0) * latent.size(2)
        else:
            sum_  += torch.sum(latent, dim=(0, 2))
            sum_sq  += torch.sum(latent**2, dim=(0, 2))
            n_samples += latent.size(0) * latent.size(2)
    mean_per_channel = sum_ / n_samples
    mean_sq_per_channel = sum_sq / n_samples
    std_per_channel = torch.sqrt(mean_sq_per_channel - mean_per_channel ** 2)
    return mean_per_channel, std_per_channel

def a_weight(fs, n_fft, min_db=-80.0):
    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = np.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (2 * np.log10(12194) + 2 * np.log10(freq_sq)
                           - np.log10(freq_sq + 12194 ** 2)
                           - np.log10(freq_sq + 20.6 ** 2)
                           - 0.5 * np.log10(freq_sq + 107.7 ** 2)
                           - 0.5 * np.log10(freq_sq + 737.9 ** 2))
    weight = np.maximum(weight, min_db)

    return weight


def compute_gain(sound, fs, min_db=-80.0, mode="A_weighting"):
    if fs == 16000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    elif fs == 48000:
        n_fft = 4096
    else:
        raise Exception("Invalid fs {}".format(fs))
    stride = n_fft // 2

    gain = []
    for i in range(0, len(sound) - n_fft + 1, stride):
        if mode == "RMSE":
            g = np.mean(sound[i: i + n_fft] ** 2)
        elif mode == "A_weighting":
            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])
            power_spec = np.abs(spec) ** 2
            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
            g = np.sum(a_weighted_spec)
        else:
            raise Exception("Invalid mode {}".format(mode))
        gain.append(g)

    gain = np.array(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)
    return gain_db


def mix(sound1, sound2, r, fs):
    gain1 = np.max(compute_gain(sound1, fs))  # Decibel
    gain2 = np.max(compute_gain(sound2, fs))
    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
    sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))
    return sound

def normalize_wav(waveform):
    waveform = waveform - torch.mean(waveform)
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-12)
    return waveform * 0.5


def pad_wav(waveform, segment_length):
    waveform_length = len(waveform)
    
    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    else:
        pad_wav = torch.zeros(segment_length - waveform_length).to(waveform.device)
        waveform = torch.cat([waveform, pad_wav])
        return waveform
    
    
def _pad_spec(fbank, target_length=1024):
    batch, n_frames, channels = fbank.shape
    p = target_length - n_frames
    if p > 0:
        pad = torch.zeros(batch, p, channels).to(fbank.device)
        fbank = torch.cat([fbank, pad], 1)
    elif p < 0:
        fbank = fbank[:, :target_length, :]

    if channels % 2 != 0:
        fbank = fbank[:, :, :-1]

    return fbank


def read_wav_file(filename, segment_length, target_sample_rate):
    waveform, sr = torchaudio.load(filename)  # Faster!!!
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sample_rate)[0]
    try:
        waveform = normalize_wav(waveform)
    except:
        print ("Exception normalizing:", filename)
        waveform = torch.ones(segment_length * target_sample_rate)
    waveform = pad_wav(waveform, segment_length).unsqueeze(0)
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = 0.5 * waveform
    return waveform


def get_mel_from_wav(audio, _stft):
    audio = torch.nan_to_num(torch.clip(audio, -1, 1))
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, log_magnitudes_stft, energy = _stft.mel_spectrogram(audio)
    return melspec, log_magnitudes_stft, energy


def wav_to_fbank(paths, target_length=1024, fn_STFT=None):
    assert fn_STFT is not None

    waveform = torch.cat([read_wav_file(path, target_length * 160) for path in paths], 0)  # hop size is 160

    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)
    fbank = fbank.transpose(1, 2)
    log_magnitudes_stft = log_magnitudes_stft.transpose(1, 2)

    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )

    return fbank, log_magnitudes_stft, waveform


def uncapitalize(s):
    if s:
        return s[:1].lower() + s[1:]
    else:
        return ""

    
def mix_wavs_and_captions(path1, path2, caption1, caption2, target_sample_rate=44100):
    sound1 = path1.cpu().numpy()
    sound2 =path2.cpu().numpy()
    mixed_sound = mix(sound1, sound2, 0.5, target_sample_rate).reshape(1, -1)
    mixed_caption = "{} and {}".format(caption1, uncapitalize(caption2))
    return mixed_sound, mixed_caption

def mix_wavs_and_audioembeds(path1, path2, embeds1, embeds2, target_sample_rate=44100):
    sound1 = path1.cpu().numpy()
    sound2 =path2.cpu().numpy()
    mixed_sound = mix(sound1, sound2, 0.5, target_sample_rate).reshape(1, -1)
    
    emb1 = embeds1.cpu().numpy()
    emb2 =embeds2.cpu().numpy()
    mixed_clap_sound = mix(emb1, emb2, 0.5, 48000).reshape(1, -1)
    return mixed_sound, mixed_clap_sound


def augment(paths, texts, num_items=4, target_sample_rate=44100):
    mixed_sounds, mixed_captions = [], []
    combinations = list(itertools.combinations(list(range(len(texts))), 2))
    random.shuffle(combinations)
    if len(combinations) < num_items:
        selected_combinations = combinations
    else:
        selected_combinations = combinations[:num_items]
        
    for (i, j) in selected_combinations:
        new_sound, new_caption = mix_wavs_and_captions(paths[i], paths[j], texts[i], texts[j], target_sample_rate)
        mixed_sounds.append(new_sound)
        mixed_captions.append(new_caption)
        
    waveform = torch.tensor(np.concatenate(mixed_sounds, 0))
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = 0.5 * waveform
    
    return waveform, mixed_captions

def augment_for_clap(paths, texts, num_items=4, target_sample_rate=44100):
    mixed_sounds, mixed_clap_sounds = [], []
    combinations = list(itertools.combinations(list(range(len(texts))), 2))
    random.shuffle(combinations)
    final_comb = []
    while len(final_comb) < num_items:
        combinations = list(itertools.combinations(list(range(len(texts))), 2))
        random.shuffle(combinations)
        for (i, j) in combinations:    
            if torch.all(texts[i] == 0.) and torch.all(texts[j] == 0.):
                continue
            else:
                final_comb.append((i, j))
            
            if len(final_comb) >= num_items:
                break
    
    final_comb = final_comb[:num_items]
    for (i, j) in final_comb:
        new_sound, new_clap_sound = mix_wavs_and_audioembeds(paths[i], paths[j], texts[i], texts[j], target_sample_rate)
        mixed_sounds.append(new_sound)
        mixed_clap_sounds.append(new_clap_sound)
                
    waveform = torch.tensor(np.concatenate(mixed_sounds, 0))
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = 0.5 * waveform

    waveform_clap = torch.tensor(np.concatenate(mixed_clap_sounds, 0))
    waveform_clap = waveform_clap / torch.max(torch.abs(waveform_clap))
    waveform_clap = 0.5 * waveform_clap
    
    return waveform, waveform_clap


def augment_wav_to_fbank(paths, texts, num_items=4, target_length=1024, fn_STFT=None):
    assert fn_STFT is not None
    
    waveform, captions = augment(paths, texts)
    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)
    fbank = fbank.transpose(1, 2)
    log_magnitudes_stft = log_magnitudes_stft.transpose(1, 2)

    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )

    return fbank, log_magnitudes_stft, waveform, captions

class PadCrop(nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output

class PadCrop_Normalized_T(nn.Module):
    
    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize

    def __call__(self, source: torch.Tensor) -> Tuple[torch.Tensor, float, float, int, int]:
        
        n_channels, n_samples = source.shape
        
        # If the audio is shorter than the desired length, pad it
        upper_bound = max(0, n_samples - self.n_samples)
        
        # If randomize is False, always start at the beginning of the audio
        offset = 0
        if(self.randomize and n_samples > self.n_samples):
            offset = random.randint(0, upper_bound)

        # Calculate the start and end times of the chunk
        t_start = offset / (upper_bound + self.n_samples)
        t_end = (offset + self.n_samples) / (upper_bound + self.n_samples)

        # Create the chunk
        chunk = source.new_zeros([n_channels, self.n_samples])

        # Copy the audio into the chunk
        chunk[:, :min(n_samples, self.n_samples)] = source[:, offset:offset + self.n_samples]
        
        # Calculate the start and end times of the chunk in seconds
        seconds_start = math.floor(offset / self.sample_rate)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        padding_mask = torch.zeros([self.n_samples])
        padding_mask[:min(n_samples, self.n_samples)] = 1
        
        
        return (
            chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total,
            padding_mask
        )

class PhaseFlipper(nn.Module):
    "Randomly invert the phase of a signal"
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def __call__(self, signal):
        return -signal if (random.random() < self.p) else signal
        
class Mono(nn.Module):
  def __call__(self, signal):
    return torch.mean(signal, dim=0, keepdims=True) if len(signal.shape) > 1 else signal

class Stereo(nn.Module):
  def __call__(self, signal):
    signal_shape = signal.shape
    # Check if it's mono
    if len(signal_shape) == 1: # s -> 2, s
        signal = signal.unsqueeze(0).repeat(2, 1)
    elif len(signal_shape) == 2:
        if signal_shape[0] == 1: #1, s -> 2, s
            signal = signal.repeat(2, 1)
        elif signal_shape[0] > 2: #?, s -> 2,s
            signal = signal[:2, :]    

    return signal


def _update_ema(ema_rate, ema_params, model_params):
    for rate, params in zip(ema_rate, ema_params):
        update_ema(params, model_params, rate=rate)

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

        
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))



def download_checkpoint(url, local_path):
    if os.path.exists(local_path):
        print(f"File already exists: {local_path}")
        return
    print(f"Downloading {url} to {local_path} ...")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(local_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download completed.")

def auto_download_checkpoints(args, train_args):
    variant_lower = args.variants.lower()
    if "v1" in variant_lower:
        version_str = "v1"
    elif "v2" in variant_lower:
        version_str = "v2"
    else:
        version_str = "v1"
    train_args["version"] = version_str

    base_url = "https://huggingface.co/koichisaito/soundctm_dit/resolve/main/"
    print(f"Using base URL: {base_url}")

    if version_str == "v1":
        ema_file = "ac_v1_iclr_ema_0999_010446.pt"
        progress_file = "ac_v1_iclr_progress_state.pth"
        summary_file = "ac_v1_iclr_summary.jsonl"
        clap_text_branch_projection = True
        z_stats_file = "z_stats.pth"
    elif version_str == "v2":
        ema_file = "ac_v2_ema_0999_030000.pt"
        progress_file = "ac_v2_progress_state.pth"
        summary_file = "ac_v2_summary.jsonl"
        clap_text_branch_projection = False
        z_stats_file = "z_stats.pth"
    vae_file = "gaussiandac/weights.pth"
    clap_file = "630k-audioset-best.pt"
    model_ckpt_path = args.model_ckpt_path
    util_ckpt_path = args.util_ckpt_path
    
    ema_local_path = os.path.join(model_ckpt_path, ema_file)
    progress_local_path = os.path.join(model_ckpt_path, progress_file)
    summary_local_path = os.path.join(model_ckpt_path, summary_file)
    z_stats_local_path = os.path.join(model_ckpt_path, z_stats_file)
    vae_local_path = os.path.join(util_ckpt_path, vae_file)
    clap_local_path = os.path.join(util_ckpt_path, clap_file)
    
    download_checkpoint(urljoin(base_url, "model_checkpoints/soundctm/" + ema_file), ema_local_path)
    download_checkpoint(urljoin(base_url, "model_checkpoints/soundctm/" + progress_file), progress_local_path)
    download_checkpoint(urljoin(base_url, "model_checkpoints/soundctm/" + summary_file), summary_local_path)
    download_checkpoint(urljoin(base_url, "model_checkpoints/soundctm/" + z_stats_file), z_stats_local_path)
    download_checkpoint(urljoin(base_url, "utils_checkpoints/clap/" + clap_file), clap_local_path)
    download_checkpoint(urljoin(base_url, "utils_checkpoints/vae/" + vae_file), vae_local_path)

    args.ema_model = ema_local_path
    args.training_args = summary_local_path
    args.load_mean_std_state_path = z_stats_local_path
    args.stage1_path = args.util_ckpt_path
    args.clap_model_path = clap_local_path

    train_args["diffusion_model_type"] = args.diffusion_model_type
    train_args["clap_model_path"] = args.clap_model_path
    train_args["stage1_path"] = args.stage1_path
    train_args["version"] = version_str
    train_args["clap_text_branch_projection"] = clap_text_branch_projection


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def read_from_config(config_path, parser):
    with open(config_path, 'r') as config_file:
        config_data = yaml.safe_load(config_file)["Experiment"]

    recognized_options = set(parser._option_string_actions.keys())
    config_args = []
    for arg_name, arg_value in config_data.items():
        option = f"--{arg_name}"
        if option not in recognized_options:
            continue
        if isinstance(arg_value, bool):
            if arg_value:
                config_args.append(option)
        elif isinstance(arg_value, list):
            config_args.append(f"{option}={str(arg_value)}")
        else:
            config_args.append(f"{option}={str(arg_value)}")
    return config_args


def rand_fix(seed):
    """Fixes random seed for reproductivity"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True