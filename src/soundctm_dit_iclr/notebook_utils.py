import os
import json
import torch
import argparse
import torchaudio
from pathlib import Path
from accelerate.utils import set_seed
from sa_edm.util import auto_download_checkpoints, rand_fix, read_from_config
from ctm.inference_sampling import karras_sample
from ctm.script_util import create_model_and_diffusion
from sa_edm.models_edm import build_stage1_models

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def parse_args(input_args=None):
    """
    If 'input_args' is None, argparse will read from sys.argv (usual CLI).
    If 'input_args' is a list, it will parse from that list (useful in notebooks).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', type=str, default='ac_v2',
                        choices=['ac_v1_iclr', 'ac_v2', 'acas_v2'],
                        help='Pretrained model variant')
    parser.add_argument('--config', type=str,
                        default="configs/hyperparameter_config/inference/dit_demo.yaml",
                        help='Path to the YAML configuration file')
    parser.add_argument('--prompt', type=str, default='bird charping near the ocean',
                        help='Input prompt')
    parser.add_argument('--cfg', type=float, default=5.,
                        help='CFG guidance weight')
    parser.add_argument('--num_steps', type=int, default=4,
                        help='Number of sampling steps')
    parser.add_argument('--nu', type=float, default=1.0,
                        help='nu parameter')
    parser.add_argument('--output_dir', type=Path, default=Path('./output'),
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=5031,
                        help='Random seed')
    parser.add_argument('--sampler', type=str, default='deterministic',
                        choices=['deterministic', 'cm_multistep', 'gamma_multistep'])
    parser.add_argument('--sampling_gamma', type=float, default=0.,
                        help='Gamma value for sampling')
    parser.add_argument('--test_file', type=str, default="data/test.csv")
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--training_args', type=str, default='configs/dummy_summary.jsonl',
                        help='Path to training summary JSONL')
    parser.add_argument('--model_ckpt_path', type=str, default='ckpt/models/')
    parser.add_argument('--util_ckpt_path', type=str, default='ckpt/utils/')
    parser.add_argument('--diffusion_model_type', type=str, default='dit',
                        help='Diffusion model type')
    parser.add_argument('--clap_text_branch_projection', action="store_false",
                        help='CLAP text branch projection config')

    if input_args is None:
        partial_args, remaining_args = parser.parse_known_args()
    else:
        partial_args, remaining_args = parser.parse_known_args(input_args)

    if not partial_args.config or partial_args.config == "":
        final_args = partial_args
        final_args = parser.parse_args(remaining_args, namespace=final_args)
    else:
        config_path = os.path.abspath(partial_args.config)
        if os.path.isfile(config_path):
            config_args_list = read_from_config(config_path, parser)
            config_args = parser.parse_args(config_args_list)
            for k, v in vars(partial_args).items():
                default_val = parser.get_default(k)
                if v != default_val:
                    setattr(config_args, k, v)
            final_args = parser.parse_args(remaining_args, namespace=config_args)
        else:
            raise FileNotFoundError(f'Configuration file "{config_path}" not found.')

    return final_args

def init_model(args):
    # Set seeds
    if args.seed is not None:
        set_seed(args.seed)
        rand_fix(args.seed)

    # Pick device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using GPU...")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU...")

    # Load training_args
    with open(args.training_args, "r", encoding="utf-8") as f:
        train_args_json = f.readlines()[0]  # or handle multi-line if needed
    train_args = DotDict(json.loads(train_args_json))

    # Auto-download checkpoints if needed
    auto_download_checkpoints(args, train_args)

    print("------------------------------------------------------------------")
    print("Initialization config:", vars(args))
    print("------------------------------------------------------------------")

    # Load the model + diffusion
    model, diffusion = create_model_and_diffusion(train_args, teacher=False)
    if not hasattr(train_args, "ema_model"):
        raise ValueError("train_args is missing 'ema_model' key.")
    model.load_state_dict(torch.load(args.ema_model, map_location=device))
    model.to(device)
    model.eval()

    # Load VAE
    if not hasattr(train_args, "stage1_path"):
        raise ValueError("train_args is missing 'stage1_path' key.")
    vae = build_stage1_models(ckpt_folder_path=args.stage1_path).to(device)
    vae.requires_grad_(False)
    vae.eval()

    # Load z_stats
    if not hasattr(train_args, "load_mean_std_state_path"):
        raise ValueError("train_args missing 'load_mean_std_state_path'.")
    progress_state = torch.load(args.load_mean_std_state_path, map_location=device)
    z_mean = progress_state['z_mean'].to(device)
    z_std = progress_state['z_std'].to(device)

    print("------------------------------------------------------------------")
    print("Model & VAE loaded successfully.")
    print("------------------------------------------------------------------")
    return {
        "device": device,
        "model": model,
        "diffusion": diffusion,
        "vae": vae,
        "z_mean": z_mean,
        "z_std": z_std,
        "train_args": train_args
    }

def run_inference(
    model_refs: dict,
    prompt: str,
    num_samples: int,
    sampler: str,
    num_steps: int,
    cfg: float,
    nu: float,
    sampling_gamma: float,
    seed: int = None
):
    device = model_refs["device"]
    model = model_refs["model"]
    diffusion = model_refs["diffusion"]
    vae = model_refs["vae"]
    z_mean = model_refs["z_mean"]
    z_std = model_refs["z_std"]
    train_args = model_refs["train_args"]

    if seed is not None:
        set_seed(seed)
        rand_fix(seed)

    print("----------------------------------------------------------")
    print("Sampling method:", sampler)
    print("Steps:", num_steps)
    print("CFG:", cfg)
    print("nu:", nu)
    print("Prompt:", prompt)
    print("----------------------------------------------------------")

    text_prompts = [prompt] * num_samples

    with torch.no_grad():
        latents = karras_sample(
            diffusion=diffusion,
            model=model,
            shape=(num_samples, 64, 432),
            steps=num_steps,
            cond=text_prompts,
            nu=nu,
            model_kwargs={},
            device=device,
            omega=cfg,
            sampler=sampler,
            gamma=sampling_gamma,
            x_T=None,
            sigma_min=train_args.sigma_min,
            sigma_max=train_args.sigma_max,
        )

        # Denormalize
        denormalized_latents = 2 * z_std[None, :, None] * latents + z_mean[None, :, None]
        waveforms = vae.decode_to_waveform(denormalized_latents)[:, :, :int(10.031 * 44100)]

    return waveforms  # shape = [num_samples, channels, samples]


def save_waveforms(
    waveforms: torch.Tensor,
    output_dir: Path,
    output_subdir: str,
    prompt: str,
    sample_rate: int = 44100
):
    """
    Saves waveforms to a subdirectory of `args.output_dir`:
      variants_{args.variants}_seed_{args.seed}_steps_{args.num_steps}_cfg_{args.cfg}_nu_{args.nu}
    Filenames are {prompt}_{index}.wav (with spaces replaced by underscores).
    
    Parameters
    ----------
    waveforms : torch.Tensor
        A batch of audio waveforms, shape [batch, channels, samples].
    args : Namespace or DotDict
        Should contain attributes: variants, seed, num_steps, cfg, nu, output_dir, etc.
    prompt : str
        The text prompt used for generation.
    sample_rate : int
        Audio sample rate used for saving. Default: 44100.
    """
    # Build the subdirectory name
    save_dir = os.path.join(str(output_dir), str(output_subdir))

    # Create directories if needed
    os.makedirs(save_dir, exist_ok=True)

    # Loop over each waveform in the batch
    for j, wav in enumerate(waveforms):
        # Ensure shape is [channels, samples]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        elif wav.dim() == 3:
            wav = wav.squeeze(0)
        elif wav.dim() == 4:
            wav = wav.squeeze(0).squeeze(0)
        
        # Generate filename based on text prompt and index
        prompt_filename = prompt.lower().replace(" ", "_")
        filename = f"{prompt_filename}_{j}.wav"
        file_path = os.path.join(save_dir, filename)

        torchaudio.save(file_path, wav.cpu(), sample_rate=sample_rate)

    print(f"Saved {len(waveforms)} files to {save_dir}")


def main():
    args = parse_args()
    
    model_refs = init_model(args)
    waveforms = run_inference(
        model_refs=model_refs,
        prompt=args.prompt,
        num_samples=args.num_samples,
        sampler=args.sampler,
        num_steps=args.num_steps,
        cfg=args.cfg,
        nu=args.nu,
        sampling_gamma=args.sampling_gamma,
        seed=args.seed
    )

    # Save
    outdir = args.output_dir
    output_subdir = f"variants_{args.variants}_seed_{args.seed}_steps_{args.num_steps}_cfg_{args.cfg}_nu_{args.nu}"
    save_waveforms(waveforms, outdir, output_subdir, prompt=args.prompt)

if __name__ == "__main__":
    main()
