import numpy as np
from sa_edm.models_edm import SA_EDM
from ctm.resample import create_named_schedule_sampler
from .karras_diffusion import KarrasDenoiser

def create_model_and_diffusion(args, teacher=False):
    schedule_sampler = create_named_schedule_sampler(args, args.schedule_sampler, args.start_scales) 
    diffusion_schedule_sampler = create_named_schedule_sampler(args, args.diffusion_schedule_sampler, args.start_scales) 
    model = SA_EDM(
        args,
        args.text_encoder_name,
        args.ctm_unet_model_config, 
        args.freeze_text_encoder, 
        teacher=teacher
    )
    
    diffusion = KarrasDenoiser(
        args=args, schedule_sampler=schedule_sampler,
        diffusion_schedule_sampler=diffusion_schedule_sampler,
    )
    
    return model, diffusion

def create_ema_and_scales_fn(
    target_ema_mode,
    start_ema, 
    scale_mode,
    start_scales,
    end_scales,
    total_steps,
    distill_steps_per_iter,
):
    def ema_and_scales_fn(step):
        if target_ema_mode == "fixed" and scale_mode == "fixed":
            target_ema = start_ema
            scales = start_scales
        elif target_ema_mode == "fixed" and scale_mode == "progressive":
            target_ema = start_ema
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            scales = scales + 1

        elif target_ema_mode == "adaptive" and scale_mode == "progressive":
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            c = -np.log(start_ema) * start_scales
            target_ema = np.exp(-c / scales)
            scales = scales + 1
        elif target_ema_mode == "fixed" and scale_mode == "progdist":
            distill_stage = step // distill_steps_per_iter
            scales = start_scales // (2**distill_stage)
            scales = np.maximum(scales, 2)

            sub_stage = np.maximum(
                step - distill_steps_per_iter * (np.log2(start_scales) - 1),
                0,
            )
            sub_stage = sub_stage // (distill_steps_per_iter * 2)
            sub_scales = 2 // (2**sub_stage)
            sub_scales = np.maximum(sub_scales, 1)

            scales = np.where(scales == 2, sub_scales, scales)

            target_ema = 1.0
        else:
            raise NotImplementedError

        return float(target_ema), int(scales)

    return ema_and_scales_fn


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}
