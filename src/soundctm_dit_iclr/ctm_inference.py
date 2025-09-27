"""This script generates samples using trained student model"""
import csv
import json
import os
import random
import time
import torchaudio
import numpy as np
import torch
from accelerate.utils import set_seed
from ctm.inference_sampling import karras_sample
from ctm.script_util import create_model_and_diffusion
from tqdm import tqdm
from sa_edm.models_edm import build_stage1_models
from sa_edm.inference_args_config import parse_args

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def rand_fix(seed):
    """Fixes random seed for reproductivity"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using GPU...")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU...")
    
    args = parse_args()
    

    if args.seed is not None:
        set_seed(args.seed)
        rand_fix(args.seed)
    
    train_args = DotDict(json.loads(open(args.training_args, encoding="utf-8").readlines()[0]))

    # print("------------------------------------------------------------------")
    # print("Inference config: ", vars(args))
    # print("Training args: ", train_args)
    # print("------------------------------------------------------------------")
    
    if args.diffusion_model_type == "dit":
        train_args["diffusion_model_type"] = args.diffusion_model_type
        train_args["dit_model_config"] = args.dit_model_config
    
    train_args["clap_model_path"] = args.clap_model_path
    train_args["stage1_path"] = args.stage1_path
    train_args["version"] = args.version
    train_args["clap_text_branch_projection"] = args.clap_text_branch_projection
    model, diffusion = create_model_and_diffusion(train_args, teacher=False) 

    model.load_state_dict(torch.load(args.ema_model, map_location=device))
    model.to(device)
    model.eval()

    print("Loading stage1 model")
    vae = build_stage1_models(ckpt_folder_path=args.stage1_path).to(device)
    vae.requires_grad_(False)
    vae.eval()

    progress_state = torch.load(os.path.join(args.load_mean_std_state_path))
    z_mean = progress_state['z_mean'].to(device)
    z_std = progress_state['z_std'].to(device)
    
    if args.prefix:
        prefix = args.prefix
    else:
        prefix = ""
    with open(args.test_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        text_prompts = [row['caption'] for row in reader]
    text_prompts = [prefix + inp for inp in text_prompts]
    with open(args.test_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        file_names = [row['file_name'] for row in reader]
    
    # Generate samples for each text caption in Test dataset
    num_steps, nu, batch_size, num_samples = args.num_steps, args.nu, args.batch_size, args.num_samples
    all_outputs = []

    print("----------------------------------------------------------")
    print("Inference configuration: ")
    print("NU: ", nu)
    print("omege: ", args.omega)
    print("steps: ", args.num_steps)
    print("sampling method: ", args.sampler)
    print("------------------------------------------------------")

    time_per_sample = []
    print(len(text_prompts))
    for k in tqdm(range(0, len(text_prompts), batch_size)):
        text = text_prompts[k: k+batch_size]
        # print("Text: ", text)
        start = time.time()
        with torch.no_grad():
            latents = karras_sample(
                diffusion=diffusion,
                model=model,
                shape=(batch_size, 64, 432),
                steps=num_steps,
                cond=text,
                nu=nu,
                model_kwargs={},
                device=device,
                omega=args.omega,
                sampler=args.sampler,
                gamma=args.sampling_gamma,
                x_T=None,
                sigma_min=train_args.sigma_min,
                sigma_max=train_args.sigma_max,
            )
        end = time.time()
        time_per_sample.append(end - start)
        denormalized_latents = 2 * z_std[None, :, None] * latents + z_mean[None, :, None]
        waveform = vae.decode_to_waveform(denormalized_latents)[:, :, :int(10.031 * 44100)]
        all_outputs += [item for item in waveform]
    
    # print("Average inference time over 957 samples : ", sum(time_per_sample) / len(time_per_sample))
    # print("Best inference time = ", min(time_per_sample))
    # print("Worst inference time = ", max(time_per_sample))
    
    # Save generated samples 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if num_samples == 1:
        output_dir = f"steps_{num_steps}_omega_{args.omega}_nu_{nu}"
        save_dir = os.path.join(args.output_dir, output_dir)
        os.makedirs(save_dir, exist_ok=True)
        for j, wav in enumerate(all_outputs):
            if wav.dim() == 1:
                wav = wav.unsqueeze(0) 
            elif wav.dim() == 3:
                wav = wav.squeeze(0)
            elif wav.dim() == 4:
                wav = wav.squeeze(0).squeeze(0) 
            torchaudio.save(f'{save_dir}/{os.path.basename(file_names[j])}', wav.to('cpu'), sample_rate=44100)

if __name__ == "__main__":
    main()