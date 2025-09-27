import json
import os
import torchaudio
import torch
from accelerate.utils import set_seed
from ctm.inference_sampling import karras_sample
from ctm.script_util import create_model_and_diffusion
from sa_edm.models_edm import build_stage1_models
from pathlib import Path
from argparse import ArgumentParser
from sa_edm.util import auto_download_checkpoints, rand_fix, read_from_config

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--variants', type=str, default='ac_v2', choices=['ac_v1_iclr', 'ac_v2', 'ac_as_v2'], help='Pretrained model variant')
    parser.add_argument('--prompt', type=str, default='wind blowing and water splashing', help='Input prompt')
    parser.add_argument('--cfg', type=float, default=5., help='CFG guidance weight')
    parser.add_argument('--num_steps', type=int, default=4, help='Number of sampling steps')
    parser.add_argument('--nu', type=float, default=1.0, help='nu parameter')
    parser.add_argument('--output_dir', type=Path, default='./output', help='Output directory')
    parser.add_argument('--seed', type=int, default=5031, help='Random seed')
    parser.add_argument('--sampler', type=str, default='deterministic', choices=['deterministic', 'cm_multistep', 'gamma_multistep'])
    parser.add_argument('--sampling_gamma', type=float, default=0., help='Gamma value for sampling')
    parser.add_argument('--test_file', type=str, default="data/test.csv")
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--training_args', type=str, default='configs/dummy_summary.jsonl', help='Path to training summary JSONL')
    parser.add_argument('--model_ckpt_path', type=str, default='ckpt/models/')
    parser.add_argument('--util_ckpt_path', type=str, default='ckpt/utils/')
    parser.add_argument('--diffusion_model_type', type=str, default='dit', help='Diffusion model type')
    parser.add_argument('--clap_text_branch_projection', action="store_false", help='CLAP text branch projection config')
    parser.add_argument('--config', type=str, default="configs/hyperparameter_config/inference/dit_demo.yaml" ,help='Path to the YAML configuration file')
    partial_args, remaining_args = parser.parse_known_args()
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

def main():
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)
        rand_fix(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using GPU...")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU...")
    
    train_args = DotDict(json.loads(open(args.training_args, encoding="utf-8").readlines()[0]))
    
    # Download checkpoints
    auto_download_checkpoints(args, train_args)

    print("------------------------------------------------------------------")
    print(f"Inference config: ", vars(args))
    print("------------------------------------------------------------------")
    
    # Load models
    model, diffusion = create_model_and_diffusion(train_args, teacher=False) 
    model.load_state_dict(torch.load(args.ema_model, map_location=device))
    model.to(device)
    model.eval()

    # Load VAE
    vae = build_stage1_models(ckpt_folder_path=args.stage1_path).to(device)
    vae.requires_grad_(False)
    vae.eval()

    # Load z_stats
    progress_state = torch.load(os.path.join(args.load_mean_std_state_path))
    z_mean = progress_state['z_mean'].to(device)
    z_std = progress_state['z_std'].to(device)
    
    
    text_prompts = [args.prompt] * args.num_samples
    # Generate samples for each text caption in Test dataset
    all_outputs = []
    
    print("----------------------------------------------------------")
    print(f"Inference configuration: ")
    print(f"sampling method: ", args.sampler)
    print(f"steps: ", args.num_steps)
    print(f"cfg: ", args.cfg)
    print(f"nu: ", args.nu)
    print(f"Start sampling...")
    print("------------------------------------------------------")

    with torch.no_grad():
        latents = karras_sample(
            diffusion=diffusion,
            model=model,
            shape=(args.num_samples, 64, 432),
            steps=args.num_steps,
            cond=text_prompts,
            nu=args.nu,
            model_kwargs={},
            device=device,
            omega=args.cfg,
            sampler=args.sampler,
            gamma=args.sampling_gamma,
            x_T=None,
            sigma_min=train_args.sigma_min,
            sigma_max=train_args.sigma_max,
        )
        denormalized_latents = 2 * z_std[None, :, None] * latents + z_mean[None, :, None]
        waveform = vae.decode_to_waveform(denormalized_latents)[:, :, :int(10.031 * 44100)]
        all_outputs += [item for item in waveform]
    
    # Save generated samples 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    output_dir = f"variants_{args.variants}_seed_{args.seed}_steps_{args.num_steps}_cfg_{args.cfg}_nu_{args.nu}"
    save_dir = os.path.join(args.output_dir, output_dir)
    os.makedirs(save_dir, exist_ok=True)
    print("save_dir", save_dir)
    for j, wav in enumerate(all_outputs):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0) 
        elif wav.dim() == 3:
            wav = wav.squeeze(0)
        elif wav.dim() == 4:
            wav = wav.squeeze(0).squeeze(0) 
    
        prompt_filename = text_prompts[j].lower().replace(" ", "_")
        filename = f"{prompt_filename}_{j}.wav"
        torchaudio.save(f'{save_dir}/{filename}', wav.to('cpu'), sample_rate=44100)

if __name__ == "__main__":
    main()