import os
import torch as th
import wandb
from tqdm import tqdm
import subprocess
import csv
import torchaudio
from ctm.inference_sampling import karras_sample
from evaluate_metrics import evaluate_generated
import time

@th.no_grad()
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

class TrainLoop:
    def __init__(
        self,
        model,
        diffusion,
        data,
        accelerator,
        opt, 
        resume_epoch=0,
        resume_step=0,
        resume_global_step=0,
        args=None,
    ):
        self.args = args
        self.accelerator = accelerator
        self.model = model
        self.diffusion = diffusion # KarrasDenoiser
        
        self.train_dataloader = data
        self.batch_size = args.per_device_train_batch_size
        self.lr = args.lr        
        self.step = 0
        self.global_step = 0
        self.first_epoch = 0
        self.resume_epoch = resume_epoch
        self.resume_step = resume_step
        self.resume_global_step = resume_global_step
        self.global_batch = self.batch_size * self.accelerator.num_processes

        self.x_T = th.randn(*(self.batch_size, 
                              self.args.latent_channels, 
                              self.args.latent_t_size, 
                              self.args.latent_f_size), 
                            device=self.accelerator.device) * self.args.sigma_max

        self.opt = opt
        self.first_epoch = self.resume_epoch
        self.step = self.resume_step
        self.global_step = self.resume_global_step

    def run_loop(self):
        while not self.args.lr_anneal_steps or self.step < self.args.lr_anneal_steps:
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.args.save_interval == 0:
                self.save()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
        if (self.step - 1) % self.args.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self.step += 1
            self._update_ema()
        self._anneal_lr()

    def forward_backward(self, batch, cond):
        raise NotImplementedError

    def _anneal_lr(self):
        if not self.args.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.args.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

class CTMTrainLoop(TrainLoop):
    def __init__(
        self,
        *,
        target_model,
        teacher_model,
        latent_decoder,
        ema_scale_fn,
        z_mean = None,
        z_std = None,
        # lr_scheduler=None, 
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.training_mode = self.args.training_mode
        self.ema_scale_fn = ema_scale_fn
        self.target_model = target_model
        self.teacher_model = teacher_model
        self.latent_decoder = latent_decoder
        self.z_mean = z_mean
        self.z_std = z_std
        # self.lr_scheduler = lr_scheduler
        self.total_training_steps = self.args.total_training_steps
        
        if teacher_model:
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()
        
        self.best_fad = 100
        self.curr_fad = self.best_fad
        self.load_eval_csv()
    
    def load_eval_csv(self):
        with open(self.args.test_file, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            text_prompts = [row['caption'] for row in reader]
            self.text_prompts = [self.args.prefix + inp for inp in text_prompts]
        
        with open(self.args.test_file, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            self.file_names = [row['file_name'] for row in reader]

    def run_loop(self):
        # save_interval = 2
        for epoch in range(self.first_epoch, self.args.num_train_epochs):
            progress_bar = tqdm(range(len(self.train_dataloader)), disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch: {epoch}")
            
            for step, batch in enumerate(self.train_dataloader):
                text, audios, _ = batch

                if self.global_step == 0:
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.sync_gradients:
                        self.save(epoch)
                    self.accelerator.wait_for_everyone()
                                
                self.run_step(audios, text)


                if (self.global_step and self.args.save_interval != -1 
                    and self.global_step % self.args.save_interval == 0 ):
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.sync_gradients:
                        self.save(epoch)
                    self.accelerator.wait_for_everyone()
                
                if self.global_step >= self.args.total_training_steps:
                    self.save(epoch)
                    break

                progress_bar.update(1)
            
            self.evaluate(self.target_model, self.global_step, "target_model")
            # self.evaluate(self.model, self.global_step, "student_model")
            
            # print("Evaluation done")
            # self.accelerator.wait_for_everyone()
            # # if self.curr_fad < self.best_fad:
            # if self.accelerator.sync_gradients:
            #     self.save_best(epoch, self.curr_fad)
            # #     self.best_fad = self.curr_fad
            # self.accelerator.wait_for_everyone()


    def run_step(self, batch, cond):
        
        if self.accelerator.is_main_process:
            result = {}
        
        estimate, target, x_start, waveform, prompt, t, s = self.get_samples(batch, cond)
        
        if (self.step+1) % self.args.gradient_accumulation_steps != 0:
            with self.accelerator.no_sync(self.model):

                losses = self.compute_gen_loss(estimate, target, x_start, waveform, prompt, t, s)

                if 'consistency_loss' in list(losses.keys()):
                    loss = self.args.consistency_weight * losses["consistency_loss"].mean()
                    
                    if 'denoising_loss' in list(losses.keys()):
                        loss = loss + self.args.denoising_weight * losses['denoising_loss'].mean()
        
                self.accelerator.backward(loss)

        else:
            losses = self.compute_gen_loss(estimate, target, x_start, waveform, prompt, t, s)

            if 'consistency_loss' in list(losses.keys()):
                loss = self.args.consistency_weight * losses["consistency_loss"].mean()
                
                if 'denoising_loss' in list(losses.keys()):
                    loss = loss + self.args.denoising_weight * losses['denoising_loss'].mean()
        
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                try:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.model_grad_clip_value)
                except:
                    self.accelerator.clip_grad_norm_(self.model.module.parameters(), self.args.model_grad_clip_value)
                    
            self.opt.step()
            # self.lr_scheduler.step()
            self.opt.zero_grad()
            
            if self.accelerator.sync_gradients:
                if self.target_model: 
                    self._update_target_ema()
                self.global_step += 1
                if self.accelerator.is_main_process:
                    # result["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                    result["learning_rate"] = self.opt.param_groups[0]['lr']
                    result["step"] = self.step
                    result["ctm_loss"] = losses["consistency_loss"].mean().detach().float()
                    result["lambda_ctm_loss"] = self.args.consistency_weight * result["ctm_loss"]
                    if 'denoising_loss' in list(losses.keys()):
                        result["dsm_loss"] = losses["denoising_loss"].mean().detach().float()
                        result["lambda_dsm_loss"] = self.args.denoising_weight * result["dsm_loss"]
                    else:
                        result["dsm_loss"] = 0.0
                        result["lambda_dsm_loss"] = 0.0
                    wandb.log(result)
                    self.accelerator.log(result, step=self.global_step)
                self._anneal_lr()
        self.step += 1

    def _update_target_ema(self):
        target_ema, scales = self.ema_scale_fn(self.global_step)
        with th.no_grad():
            if self.args.diffusion_model_type == "unet":
                try:
                    update_ema(
                        list(self.target_model.unet.parameters()),
                        list(self.model.unet.parameters()),
                        rate=target_ema,
                    )
                except:
                    update_ema(
                        list(self.target_model.unet.parameters()),
                        list(self.model.module.unet.parameters()),
                        rate=target_ema,
                    )
            elif self.args.diffusion_model_type == "dit":
                try:
                    update_ema(
                        list(self.target_model.dit.parameters()),
                        list(self.model.dit.parameters()),
                        rate=target_ema,
                    )
                except:
                    update_ema(
                        list(self.target_model.dit.parameters()),
                        list(self.model.module.dit.parameters()),
                        rate=target_ema,
                    )
            else:
                raise NotImplementedError
    # CHANGED    
    def get_samples(self, batch, cond):
        estimate, target, x_start, waveform, prompt, t, s = self.diffusion.get_samples(
            step = self.global_step,
            model = self.model,
            waveform = batch,
            prompt = cond,
            target_model = self.target_model,
            teacher_model = self.teacher_model,
            stage1_model = self.latent_decoder,
            accelerator = self.accelerator,
            noise=None,
            ctm = True if self.training_mode.lower() == 'ctm' else False,
            max_num_aug = 1,
            sample_rate = 44100,
            z_mean = self.z_mean,
            z_std = self.z_std
        )

        return estimate, target, x_start, waveform, prompt, t, s
    
    def compute_gen_loss(self, estimate, target, x_start, waveform, prompt, t, s):
        losses = self.diffusion.get_gen_loss(
            step = self.global_step,
            model = self.model, # self.ddp_model
            estimate = estimate,
            target = target,
            x_start = x_start,
            prompt = prompt,
            t = t,
            s = s,
            teacher_model = self.teacher_model,
        )
        
        return  losses
    
    def save(self, epoch):
        if self.accelerator.is_main_process:
            # Creating output directory
            os.makedirs(os.path.join(self.args.output_dir, f"{self.global_step:06d}"), exist_ok=True)
            
            #-----------------------------------------
            # SAVING TARGET MODEL
            #-----------------------------------------
            self.accelerator.print(f"saving target model...")
            target_state_dict = self.target_model.state_dict()
            target_file_path = os.path.join(self.args.output_dir, f"{self.global_step:06d}", f"target_{self.global_step:06d}.pt")
            self.accelerator.save(target_state_dict, target_file_path)
            
            #-----------------------------------------
            # SAVING STUDENT MODEL
            #-----------------------------------------
            self.accelerator.print("saving the student model....")
            student_state_dict = self.model.state_dict()    
            student_file_path = os.path.join(self.args.output_dir, f"{self.global_step:06d}", f"student_{self.global_step:06d}.pt")
            self.accelerator.save(student_state_dict, student_file_path)

            #-----------------------------------------
            # SAVING OPTIMIZER STATES
            #-----------------------------------------
            self.accelerator.print("saving optimizer state...")
            progress_output_dir = os.path.join(self.args.output_dir, f"{self.global_step:06d}", f"progress_state.pth")
            progress_state_dict = {
            'completed_epochs': int(epoch),
            'completed_steps': int(self.step),
            'completed_global_steps': int(self.global_step)
            }
            self.accelerator.save(progress_state_dict, progress_output_dir)
            self.accelerator.save_state("{}/{}".format(self.args.output_dir, f"{self.global_step:06d}")) 

            # ONLY FOR CRUSOE 
            self.change_owner()

    def save_best(self, epoch, fad):
        if self.accelerator.is_main_process:
            # Creating output directory
            os.makedirs(os.path.join(self.args.output_dir, "best"), exist_ok=True)
            
            #-----------------------------------------
            # SAVING TARGET MODEL
            #-----------------------------------------
            self.accelerator.print(f"saving target model...")
            target_state_dict = self.target_model.state_dict()
            target_file_path = os.path.join(self.args.output_dir, "best", f"target_{fad}_{self.global_step:06d}.pt")
            self.accelerator.save(target_state_dict, target_file_path)
            
            #-----------------------------------------
            # SAVING STUDENT MODEL
            #-----------------------------------------
            self.accelerator.print("saving the student model....")
            student_state_dict = self.model.state_dict()    
            student_file_path = os.path.join(self.args.output_dir, "best", f"student_{fad}_{self.global_step:06d}.pt")
            self.accelerator.save(student_state_dict, student_file_path)

            #-----------------------------------------
            # SAVING OPTIMIZER STATES
            #-----------------------------------------
            self.accelerator.print("saving optimizer state...")
            progress_output_dir = os.path.join(self.args.output_dir, "best", f"progress_state.pth")
            progress_state_dict = {
            'completed_epochs': int(epoch),
            'completed_steps': int(self.step),
            'completed_global_steps': int(self.global_step)
            }
            self.accelerator.save(progress_state_dict, progress_output_dir)
            self.accelerator.save_state("{}/best/".format(self.args.output_dir))

            # ONLY FOR CRUSOE 
            self.change_owner()
    
    # def save_best(self, epoch, fad):
    #     print("trying to save best........................................")
    #     rate = float(fad)
    #     if self.accelerator.is_main_process:
    #         print("saving the target model....")
    #         try:
    #             state_dict = self.target_model.state_dict()
    #         except:
    #             state_dict = self.target_model.module.state_dict()
                
    #         self.accelerator.print(f"saving model {rate}...")
    #         if not rate:
    #             filename = f"model{self.global_step:06d}.pt"
    #         else:
    #             filename = f"fad_{rate}_{self.global_step:06d}.pt"

    #         ema_output_dir = os.path.join(self.args.output_dir,"best", "target_model", filename)
            
    #         os.makedirs(os.path.join(self.args.output_dir,"best", "target_model"), exist_ok=True)
    #         self.accelerator.save(state_dict, ema_output_dir)

    #         self.accelerator.print("saving state...")
    #         progress_output_dir = os.path.join(self.args.output_dir, "best", "target_model", f"progress_state.pth")
    #         progress_state_dict = {
    #         'completed_epochs': int(epoch),
    #         'completed_steps': int(self.step),
    #         'completed_global_steps': int(self.global_step)
    #         }
    #         self.accelerator.save(progress_state_dict, progress_output_dir)
    #         self.accelerator.save_state("{}/best/target_model/".format(self.args.output_dir)) # define output dir

    #         self.change_owner()
    #         #-------------------------------------------------------------------------------------------------
    #         #  EMA state 0
    #         #-------------------------------------------------------------------------------------------------
    #         print("saving the student model....")
    #         try:
    #             state_dict = self.model.state_dict()
    #         except:
    #             state_dict = self.model.module.state_dict()
                
    #         self.accelerator.print(f"saving model {rate}...")
    #         if not rate:
    #             filename = f"model{self.global_step:06d}.pt"
    #         else:
    #             filename = f"fad_{rate}_{self.global_step:06d}.pt"
            
    #         ema_output_dir = os.path.join(self.args.output_dir, "best", "student_model", filename)
    #         os.makedirs(os.path.join(self.args.output_dir, "best", "student_model"), exist_ok=True)

    #         self.accelerator.save(state_dict, ema_output_dir)
    #         self.accelerator.print("saving state...")
            
    #         progress_output_dir = os.path.join(self.args.output_dir, "best", "student_model", f"progress_state.pth")
            
    #         progress_state_dict = {
    #         'completed_epochs': int(epoch),
    #         'completed_steps': int(self.step),
    #         'completed_global_steps': int(self.global_step)
    #         }
    #         self.accelerator.save(progress_state_dict, progress_output_dir)
    #         self.accelerator.save_state("{}/best/student_model/".format(self.args.output_dir)) # define output dir

    #         # ONLY FOR CRUSOE
    #         self.change_owner()

    def change_owner(self):
        cmd = "chown -R $USER:ct "
        subprocess.call(cmd + self.args.output_dir, shell=True)
    
    def evaluate(self, model, step, model_type):
        start = time.time()
        if self.accelerator.is_main_process:
            model.eval()
            all_outputs = []
            for k in tqdm(range(0, len(self.text_prompts), self.args.eval_batch_size)):
                text = self.text_prompts[k: k+self.args.eval_batch_size]
                with th.no_grad():
                    latents = karras_sample(
                        diffusion=self.diffusion,
                        model=model,
                        shape=(1, 64, 432),
                        steps=self.args.eval_num_steps,
                        cond=text,
                        nu=self.args.nu,
                        model_kwargs={},
                        device=self.accelerator.device,
                        omega=self.args.omega,
                        sampler=self.args.sampler,
                        gamma=self.args.sampling_gamma,
                        x_T=None,
                        sigma_min=self.args.sigma_min,
                        sigma_max=self.args.sigma_max,
                    )
                    
                denormalized_latents = 2 * self.z_std[None, :, None] * latents + self.z_mean[None, :, None]
                waveform = self.latent_decoder.decode_to_waveform(denormalized_latents)[:, :, :int(self.args.target_sec * self.args.sampling_rate)]
                all_outputs += [item for item in waveform]
            
            generation_dir = os.path.join("eval", f"tmp_{model_type}")
            if not os.path.exists(generation_dir):
                os.makedirs(generation_dir, exist_ok=True)

            for j, wav in enumerate(all_outputs):
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0) 
                elif wav.dim() == 3:
                    wav = wav.squeeze(0)
                elif wav.dim() == 4:
                    wav = wav.squeeze(0).squeeze(0) 
                torchaudio.save(f'{generation_dir}/{os.path.basename(self.file_names[j])}', wav.to('cpu'), sample_rate=self.args.sampling_rate)

            with th.no_grad():
                device = self.accelerator.device
                import torch.multiprocessing as mp
                mp.set_sharing_strategy('file_system')

                eval_args = {
                                    "reference_dir": self.args.reference_dir, 
                                    "generated_dir": generation_dir, 
                                    "reference_path_acid": self.args.test_file_ACID
                                    }
                metrics = evaluate_generated(eval_args, device=device)
                print(f"FD OpenL3_{model_type}: {metrics[0][1]} \n CLAP score_{model_type}: {metrics[1][1]} \n KL Passt score_{model_type}: {metrics[2][1]} \n FD CLAP_{model_type}: {metrics[3][1]} \n MMD CLAP score_{model_type}: {metrics[4][1]}")
            wandb.log(  {
                        f"FD OpenL3_{model_type}":       metrics[0][1],
                        f"CLAP score_{model_type}":      metrics[1][1],
                        f"KL Passt score_{model_type}":  metrics[2][1],
                        f"FD CLAP_{model_type}":         metrics[3][1],
                        f"MMD CLAP score_{model_type}":  metrics[4][1]
                        }, 
                        step=step)
        end = time.time()
        print(f"accelerator is out of if in eval time taken {end - start}")
        self.accelerator.wait_for_everyone()
