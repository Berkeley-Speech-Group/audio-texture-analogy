import torch
import torchaudio
import gc
import argparse
import os
from tqdm import tqdm
import wandb
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler, StyleVDiffusion, StyleVSampler
import sys
sys.path.append("/mnt/Datasets/soundctm_dit_iclr/")
from sa_edm.models_edm import build_stage1_models
from network import FusionPEModel
from dataset.dataset_atm import AUDIT_Dataset
import random
import laion_clap
from util import load_clap_model



def create_model():
    return DiffusionModel(
        net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
        dim=2, # for spectrogram we use 2D-CNN
        in_channels=2, # U-Net: number of input (audio) channels
        out_channels=1, # U-Net: number of output (audio) channels
        channels=[256, 512, 1024, 1024, 1024], # U-Net: channels at each layer
        factors=[2, 2, 2, 1, 1], # U-Net: downsampling and upsampling factors at each layer
        items=[2, 2, 2, 2, 2], # U-Net: number of repeating items at each layer
        attentions=[0, 0, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
        attention_heads=8, # U-Net: number of attention heads per attention item
        attention_features=64, # U-Net: number of attention features per attention item
        diffusion_t=StyleVDiffusion, # The diffusion method used
        sampler_t=StyleVSampler, # The diffusion sampler used
        use_embedding_cfg=True, # Use classifier free guidance
        embedding_max_length=2, # Maximum length of the embeddings
        embedding_features=512, # U-Net: embedding features
        cross_attentions=[0, 0, 1, 1, 1], # U-Net: cross-attention enabled/disabled at each layer 
    )

def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using GPU...")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU...")


    train_dataset = AUDIT_Dataset(
        path="/mnt/Datasets/audioset_subset/k/l"
    )

    test_dataset = AUDIT_Dataset(
        path="/mnt/Datasets/audioset_subset/k/l"

    )

    print(f"Dataset length: {len(train_dataset)}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = create_model().to(device)
    
    # load clap model
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base').to(device)
    clap_model = load_clap_model(clap_model, args.clap_ckpt)
    # clap_model.requires_grad_(False)
    print("------CLAP model loaded-----")

    # load fusion pe model    
    pe_model = FusionPEModel(512).to(device)


    # load vae
    vae = build_stage1_models(ckpt_folder_path=args.vae_ckpt_path).to(device)
    vae.requires_grad_(False)
    vae.eval()
    print("------VAE model loaded-----")

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    params = list(pe_model.parameters()) + list(clap_model.parameters())
    clap_optimizer = torch.optim.Adam(params, lr=1e-4, betas=(0.99, 0.9))
    optimizer = torch.optim.AdamW(params=list(model.parameters()), lr=1e-4, betas= (0.95, 0.999), eps=1e-6, weight_decay=1e-3)

    # print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    run_id = wandb.util.generate_id()
    if args.run_id is not None:
        run_id = args.run_id
    print(f"Run ID: {run_id}")

    wandb.init(project="ldm", resume=args.resume, id=run_id)

    epoch = 0
    step = 0

    os.makedirs(os.path.join(args.checkpoint, run_id, 'wavs'), exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint, run_id)

    if wandb.run.resumed:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path + "/latest.pt")
        else:
            checkpoint = torch.load(wandb.restore(checkpoint_path))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        clap_optimizer.load_state_dict(checkpoint['clap_optimizer_state_dict'])
        clap_model.load_state_dict(checkpoint['clap_state_dict'])
        pe_model.load_state_dict(checkpoint['pe_model_state_dict'])
        epoch = checkpoint['epoch']
        step = epoch * len(train_dataloader)
    
    # scaler = torch.cuda.amp.GradScaler()

    # Initialize the best test loss with a high value to make sure the first test_avg_loss will be considered better.
    best_test_loss = 100

    pe_model.train()
    clap_model.train()
    model.train()
    
    for epoch in range(epoch, 2500):

        avg_loss = 0
        avg_loss_step = 0
        progress = tqdm(train_dataloader, ncols=80)

        for i, batch in enumerate(progress):

            optimizer.zero_grad()
            clap_optimizer.zero_grad()
            wf_bef, wf_aft, cond_bef, cond_aft = [t.to(device) for t in batch]

            with torch.no_grad():
                latent_bef = vae.encode_to_latent(wf_bef, sample_rate=args.sampling_rate).unsqueeze(1)
                latent_aft = vae.encode_to_latent(wf_aft, sample_rate=args.sampling_rate).unsqueeze(1)
                cond_emb_bef = torch.from_numpy(clap_model.get_audio_embedding_from_data(x=cond_bef.squeeze(1).cpu().numpy())).to(device)
                cond_emb_aft = torch.from_numpy(clap_model.get_audio_embedding_from_data(x=cond_aft.squeeze(1).cpu().numpy())).to(device)
            
            cond_embed = pe_model(cond_emb_bef, cond_emb_aft)

            # print(latent_bef.shape, latent_aft.shape, cond_embed.shape)
            loss = model(latent_bef, latent_aft, embedding=cond_embed, embedding_mask_proba=0.1)
            avg_loss += loss.item()
            avg_loss_step += 1
            loss.backward()
            optimizer.step()
            clap_optimizer.step()
            progress.set_postfix(
                # loss=loss.item(),
                loss=avg_loss / avg_loss_step,
                epoch=epoch + i / len(train_dataloader),
            )


            if step % 500 == 0: # 500
                model.eval()
                pe_model.eval()
                clap_model.eval()
                test_wf_bef, test_wf_aft, test_cond_bef, test_cond_aft = [t.to(device).unsqueeze(0) for t in random.choice(test_dataset)]
                with torch.no_grad():
                    test_latent_bef = vae.encode_to_latent(test_wf_bef, sample_rate=args.sampling_rate).unsqueeze(1)
                    test_latent_aft = vae.encode_to_latent(test_wf_aft, sample_rate=args.sampling_rate).unsqueeze(1)
                    test_cond_emb_bef = torch.from_numpy(clap_model.get_audio_embedding_from_data(x=test_cond_bef.squeeze(1).cpu().numpy())).to(device)
                    test_cond_emb_aft = torch.from_numpy(clap_model.get_audio_embedding_from_data(x=test_cond_aft.squeeze(1).cpu().numpy())).to(device)
                    test_cond_embed = pe_model(test_cond_emb_bef, test_cond_emb_aft)

                    # Turn noise into new audio sample with diffusion
                    noise = torch.randn_like(test_latent_bef).to(device)
                    # with torch.cuda.amp.autocast():
                    test_latent_gen = model.sample(test_latent_bef[0], noise, embedding=test_cond_embed, num_steps=200, embedding_scale=4.5)

                    recon_wf_tgt = vae.decode_to_waveform(test_latent_aft[0])[:, :, :test_wf_bef.shape[-1]]
                    recon_wf_gen = vae.decode_to_waveform(test_latent_gen[0])[:, :, :test_wf_aft.shape[-1]]


                torchaudio.save(os.path.join(checkpoint_path, 'wavs', f"recon_wf_tgt_{step}.wav"), recon_wf_tgt[0].cpu(), args.sampling_rate)
                torchaudio.save(os.path.join(checkpoint_path, 'wavs', f"recon_wf_gen_{step}.wav"), recon_wf_gen[0].cpu(), args.sampling_rate)
                torchaudio.save(os.path.join(checkpoint_path, 'wavs', f"test_cond_bef_{step}.wav"), test_cond_bef[0].cpu(), args.sampling_rate)
                torchaudio.save(os.path.join(checkpoint_path, 'wavs', f"test_cond_aft_{step}.wav"), test_cond_aft[0].cpu(), args.sampling_rate)


                wandb.log({
                    "step": step,
                    "epoch": epoch + i / len(train_dataloader),
                    "loss": avg_loss / avg_loss_step,
                })
                model.train()
                pe_model.train()
                clap_model.train()
            
            if step % 100 == 0:
                wandb.log({
                    "step": step,
                    "epoch": epoch + i / len(train_dataloader),
                    "loss": avg_loss / avg_loss_step,
                })
                avg_loss = 0
                avg_loss_step = 0
            
            step += 1

        # Evaluate on test set
        model.eval()
        pe_model.eval()
        clap_model.eval()
        test_loss = 0
        test_loss_step = 0
        test_progress = tqdm(test_dataloader, ncols=80)

        for i, batch in enumerate(test_progress):
            test_wf_bef, test_wf_aft, test_cond_bef, test_cond_aft = [t.to(device) for t in batch]

            with torch.no_grad():
                test_latent_bef = vae.encode_to_latent(test_wf_bef, sample_rate=args.sampling_rate).unsqueeze(1)
                test_latent_aft = vae.encode_to_latent(test_wf_aft, sample_rate=args.sampling_rate).unsqueeze(1)
                test_cond_emb_bef = torch.from_numpy(clap_model.get_audio_embedding_from_data(x=test_cond_bef.squeeze(1).cpu().numpy())).to(device)
                test_cond_emb_aft = torch.from_numpy(clap_model.get_audio_embedding_from_data(x=test_cond_aft.squeeze(1).cpu().numpy())).to(device)     
                test_cond_embed = pe_model(test_cond_emb_bef, test_cond_emb_aft)
                test_loss += model(test_latent_bef, test_latent_aft, embedding=test_cond_embed, embedding_mask_proba=0.1).item()
                test_loss_step += 1
                test_avg_loss = test_loss / test_loss_step
            test_progress.set_postfix(
                test_loss=test_avg_loss,
                epoch=epoch + i / len(test_dataloader),
            )
        # if current test loss is better than previous best, save model
        epoch += 1
        if test_avg_loss < best_test_loss:
            best_test_loss = test_avg_loss
            # Save the model checkpoint with the current best test loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'pe_model_state_dict': pe_model.state_dict(),
                'clap_state_dict': clap_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'clap_optimizer_state_dict': clap_optimizer.state_dict(),
            }, os.path.join(checkpoint_path, "best.pt"))
            # Log the best_test_loss to WandB using wandb.log
            wandb.log({"best_test_loss": best_test_loss})
            wandb.save(checkpoint_path, base_path=args.checkpoint)

        # Save the latest model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'pe_model_state_dict': pe_model.state_dict(),
            'clap_state_dict': clap_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'clap_optimizer_state_dict': clap_optimizer.state_dict(),
        }, os.path.join(checkpoint_path, "latest.pt"))
        model.train()
        pe_model.train()        
        clap_model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default='/scratch/weihan/baselines/AUDIT/checkpoints/')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_id", type=str) # , default='ldm_audio_condition_no_overlap'
    parser.add_argument("--sampling_rate", type=float, default=44100, help='Sampling rate of training data')
    parser.add_argument('--vae_ckpt_path', type=str, default='/mnt/Datasets/soundctm_dit_iclr/ckpt/utils_checkpoints/vae/')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--clap_ckpt', type=str, default='/scratch/weihan/baselines/AUDIT/clap_ckpt.pt')

    return parser.parse_args()


if __name__ == "__main__":
    main()



# CUDA_VISIBLE_DEVICES=0 python /scratch/weihan/baselines/AUDIT/train_atm.py --run_id=atm