import torch
import torch.nn as nn
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.weight_norm import WeightNorm
from sa_edm.dac.model.dac import GaussianDAC
from transformers import CLIPTokenizer, AutoTokenizer
from transformers import CLIPTextModel, T5EncoderModel, AutoModel
from sa_edm.unet.unet_1d_condition import CTMUNet1DConditionModel 
from sa_edm.unet.unet_1d_condition_teacher import UNet1DConditionModel as UNet1DConditionModel_Teacher
from sa_edm.clap_modified.modified_hook import CLAP_Module_modified
from sa_edm.stable_audio_tools.models.dit import CTMDiffusionTransformer
from sa_edm.stable_audio_tools.models.dit_teacher import DiffusionTransformer as TeacherDiffusionTransformer
import json

def remove_weight_norms(self):
    print('Removing weight norm...')
    for module in self.modules():
        if isinstance(module, WeightNorm):
            remove_weight_norm(module)

def build_stage1_models(ckpt_folder_path):
    kwargs = {
        "folder": f"{ckpt_folder_path}",
        "map_location": "cpu",
        "package": False,
    }
    vae, _ = GaussianDAC.load_from_folder(**kwargs)
    vae.__class__.remove_weight_norm = remove_weight_norms
    vae.eval()
    vae.remove_weight_norm()
    
    return vae

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


class SA_EDM(nn.Module):
    def __init__(
        self,
        args,
        text_encoder_name,
        unet_model_config_path=None,
        freeze_text_encoder: bool = True,
        precond_type: str = 'edm',
        use_fp16: bool = False,
        force_fp32: bool = False,
        amodel=None,
        teacher: bool = False,
    ):
        super().__init__()

        self.text_encoder_name = text_encoder_name
        self.unet_model_config_path = unet_model_config_path
        self.freeze_text_encoder = freeze_text_encoder

        self.model_type = args.diffusion_model_type

        if teacher:
            if self.model_type == "unet":
                self.unet_config = UNet1DConditionModel_Teacher.load_config(unet_model_config_path)
                self.unet = UNet1DConditionModel_Teacher.from_config(self.unet_config, subfolder="unet")
                self.device = self.unet.device

            elif self.model_type == "dit":
                if args.dit_model_config == "":
                    print(f"Loading the default DIT configuration version selected: {args.version}")
                    if args.version == "v2":
                        dit_model_config = "configs/dit_config/dit_v2.json"
                        args.clap_text_branch_projection = False
                    elif args.version == "v1":
                        dit_model_config = "configs/dit_config/dit_v1.json"
                    else:
                        raise NotImplementedError
                else:
                    dit_model_config = args.dit_model_config
                    print(f"Loading custom DIT configuration from {args.dit_model_config}")
                
                print("Loading Diffusion Transformer Architecture...........")

                self.dit_config = json.load(open(dit_model_config))
                self.dit = TeacherDiffusionTransformer(
                                                io_channels=self.dit_config["io_channels"],
                                                embed_dim=int(self.dit_config["embed_dim"]),
                                                cond_token_dim=int(self.dit_config["cond_token_dim"]),
                                                project_cond_tokens=int(self.dit_config["project_cond_tokens"]),
                                                global_cond_dim=int(self.dit_config["global_cond_dim"]),
                                                transformer_type=self.dit_config["transformer_type"],
                                                num_heads=int(self.dit_config["num_heads"]),
                                                depth=int(self.dit_config["depth"])
                                                )
                self.device = next(self.dit.parameters()).device

            else:
                raise NotImplementedError

        if not teacher:
            if self.model_type == "unet":
                self.unet_config = CTMUNet1DConditionModel.load_config(unet_model_config_path)
                self.unet = CTMUNet1DConditionModel.from_config(self.unet_config, subfolder="unet")
                self.device = self.unet.device
            elif self.model_type == "dit":
                # if args.dit_model_config == "":
                print(f"Loading the default DIT configuration version selected: {args.version}")
                if args.version == "v2":
                    dit_model_config = "configs/dit_config/dit_v2.json"
                    args.clap_text_branch_projection = False
                elif args.version == "v1":
                    dit_model_config = "configs/dit_config/dit_v1.json"
                else:
                    raise NotImplementedError
                # else:
                #     dit_model_config = args.dit_model_config
                #     print(f"Loading custom DIT configuration from {args.dit_model_config}")
                
                print("Loading Diffusion Transformer Architecture...........")

                self.dit_config = json.load(open(dit_model_config))

                self.dit = CTMDiffusionTransformer(
                                                io_channels=self.dit_config["io_channels"],
                                                embed_dim=int(self.dit_config["embed_dim"]),
                                                cond_token_dim=int(self.dit_config["cond_token_dim"]),
                                                project_cond_tokens=int(self.dit_config["project_cond_tokens"]),
                                                global_cond_dim=int(self.dit_config["global_cond_dim"]),
                                                transformer_type=self.dit_config["transformer_type"],
                                                num_heads=int(self.dit_config["num_heads"]),
                                                depth=int(self.dit_config["depth"])
                                                )
                self.device = next(self.dit.parameters()).device

            else:
                raise NotImplementedError


        self.set_from = "random"
            
        if "stable-diffusion" in self.text_encoder_name:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.text_encoder_name, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(self.text_encoder_name, subfolder="text_encoder")
        elif "t5" in self.text_encoder_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
            self.text_encoder = T5EncoderModel.from_pretrained(self.text_encoder_name)
        elif "clap" in self.text_encoder_name:
            amodel = 'HTSAT-tiny' if amodel == "None" or amodel == None else amodel
            self.text_encoder = CLAP_Module_modified(enable_fusion=False, device = self.device, amodel=amodel, projection=args.clap_text_branch_projection)
            self.text_encoder.load_ckpt(args.clap_model_path, verbose=False)
            for param in self.text_encoder.model.parameters():
                param.requires_grad = False
            self.text_encoder.model.eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
            self.text_encoder = AutoModel.from_pretrained(self.text_encoder_name)
            
        self.dtype = torch.float16 if (use_fp16 and not force_fp32) else torch.float32
        self.text_audio_pair_dataset = args.text_audio_pair_dataset
        
    
    def encode_text(self, prompt):
        device = self.device
        batch = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        if self.freeze_text_encoder:
            with torch.no_grad():
                encoder_hidden_states = self.text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )[0]
        else:
            encoder_hidden_states = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

        boolean_encoder_mask = (attention_mask == 1).to(device)
        return encoder_hidden_states, boolean_encoder_mask

    def extract_feature_space(self, latents, timesteps=None, prompt=None, unet_mode = 'half'):        
        device = latents.device
        bsz = latents.shape[0]
        if "t5" in self.text_encoder_name:
            encoder_hidden_states, boolean_encoder_mask = self.encode_text(prompt)        
        elif "clap" in self.text_encoder_name:
            embeddings = []
            if self.text_audio_pair_dataset:
                for text in prompt:
                    embeddings.append(self.text_encoder.get_text_embedding(text, use_tensor=True).unsqueeze(0).to(device))
                encoder_hidden_states = torch.cat(embeddings, dim=0).to(device)
            else:
                for audio in prompt:
                    if torch.all(audio == 0.):
                        encoder_hidden_state = self.text_encoder.get_text_embedding("", use_tensor=True).unsqueeze(0).to(device)
                    else:
                        encoder_hidden_state = self.text_encoder.get_audio_embedding_from_data(audio, use_tensor=True).unsqueeze(0).to(device)
                    embeddings.append(encoder_hidden_state)
                encoder_hidden_states = torch.cat(embeddings, dim=0).to(device)
            boolean_encoder_mask = torch.ones([bsz, 1], dtype=torch.bool, device=device)
        
        noisy_latents = latents
        sigma = timesteps
        c_noise = sigma.log() / 4

        if self.model_type == 'unet':

            if unet_mode == 'half':
                fmaps = self.unet.get_feature(
                    noisy_latents, c_noise.squeeze(), encoder_hidden_states,
                    encoder_attention_mask=boolean_encoder_mask
                )
            elif unet_mode == 'full':
                fmaps = self.unet.get_feature_full(
                    noisy_latents, c_noise.squeeze(), encoder_hidden_states,
                    encoder_attention_mask=boolean_encoder_mask
                )

        elif self.model_type == 'dit':
            fmaps = self.dit.get_feature(noisy_latents, c_noise.squeeze(), cross_attn_cond=encoder_hidden_states, cross_attn_cond_mask=boolean_encoder_mask)
        return fmaps


    def guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=w.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype, device=w.device) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb
        
    def forward(self, latents, timesteps=None, prompt=None, s_timesteps=None, teacher=False, cfg=None, **kwargs):

        device = latents.device
        bsz = latents.shape[0]
        if "t5" in self.text_encoder_name:
            encoder_hidden_states, boolean_encoder_mask = self.encode_text(prompt)        
        elif "clap" in self.text_encoder_name:
            embeddings = []
            if self.text_audio_pair_dataset:
                for text in prompt:
                    embeddings.append(self.text_encoder.get_text_embedding(text, use_tensor=True).unsqueeze(0).to(device))
                encoder_hidden_states = torch.cat(embeddings, dim=0).to(device)
            else:
                for audio in prompt:
                    if torch.all(audio == 0.):
                        encoder_hidden_state = self.text_encoder.get_text_embedding("", use_tensor=True).unsqueeze(0).to(device)
                    else:
                        encoder_hidden_state = self.text_encoder.get_audio_embedding_from_data(audio, use_tensor=True).unsqueeze(0).to(device)
                    embeddings.append(encoder_hidden_state)
                encoder_hidden_states = torch.cat(embeddings, dim=0).to(device)
            boolean_encoder_mask = torch.ones([bsz, 1], dtype=torch.bool, device=device)
        noisy_latents = latents 

        if teacher:
            sigma = timesteps
            c_noise = sigma.log() / 4
            if self.model_type == "dit":
                F_x = self.dit._forward(noisy_latents, c_noise.squeeze(), cross_attn_cond=encoder_hidden_states, cross_attn_cond_mask=boolean_encoder_mask)
            else:
                
                F_x = self.unet(
                    noisy_latents, c_noise.squeeze(), encoder_hidden_states,
                    encoder_attention_mask=boolean_encoder_mask
                ).sample # F_x in eq(7)

        else:
            if cfg is not None:
                if self.model_type == 'dit':
                    w_embedding = self.guidance_scale_embedding(cfg, embedding_dim=self.dit_config["embed_dim"])
                    w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)
                elif self.model_type == "unet":
                    w_embedding = self.guidance_scale_embedding(cfg, embedding_dim=self.unet_config['block_out_channels'][0]*4)
                    w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)
                else:
                    raise NotImplementedError

            t = timesteps
            t = t.log() / 4

            if s_timesteps != None:
                s = s_timesteps
                s = s.log() / 4

            if self.model_type == 'dit':
                F_x = self.dit._forward(noisy_latents, 
                                            t.flatten(), 
                                            s_timestep=None if s == None else s.flatten(), 
                                            embedd_cfg=None if cfg == None else w_embedding, 
                                            cross_attn_cond=encoder_hidden_states, 
                                            cross_attn_cond_mask=boolean_encoder_mask)

            else:
                F_x = self.unet(
                    noisy_latents, 
                    t.flatten(),
                    encoder_hidden_states=encoder_hidden_states,
                    s_timestep=None if s == None else s.flatten(),
                    embedd_cfg=None if cfg == None else w_embedding,
                    encoder_attention_mask=boolean_encoder_mask
                ).sample
        

        return F_x
