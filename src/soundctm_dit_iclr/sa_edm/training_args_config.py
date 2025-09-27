import argparse
import yaml
import os
from transformers import SchedulerType

def ctm_train_defaults():
    return dict(
        consistency_weight=1.0,
        loss_type='l2',
        unet_mode = 'full',
        schedule_sampler="uniform",
        weight_schedule="uniform",
        parametrization='euler',
        inner_parametrization='edm',
        num_heun_step=39,
        num_heun_step_random=True,
        training_mode="ctm",
        match_point='z0', #
        target_ema_mode="fixed",
        scale_mode="fixed",
        start_ema=0.999, 
        start_scales=40,
        end_scales=40,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7,
        latent_channels=8,
        latent_f_size=16,
        latent_t_size=256,
        lr=0.00008,
        weight_decay=0.0,
        lr_anneal_steps=0,
        ema_rate="0.999", 
        total_training_steps=600000,
        save_interval=5000,
        cfg_single_distill=False,
        single_target_cfg=3.5,
        unform_sampled_cfg_distill=True,
        w_min=2.0,
        w_max=5.0,
        distill_steps_per_iter=50000,

        sample_s_strategy='uniform',
        heun_step_strategy='weighted',
        heun_step_multiplier=1.0,
        auxiliary_type='stop_grad',
        time_continuous=False,
        
        diffusion_training=True,
        denoising_weight=1.,
        diffusion_mult = 0.7,
        diffusion_schedule_sampler='halflognormal',
        apply_adaptive_weight=True,
        dsm_loss_target='z_0', # z_0 or z_target
        diffusion_weight_schedule="karras_weight",
    )

def ldm_defaults():
    return dict(
        target_sec = 10.031,
        sampling_rate = 44100,
        clap_model_path = "/group/ct/text_to_audio/checkpoints/clap/630k-audioset-best.pt",
        amodel = "None",
        audio_channel = 'mono',    
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7,
        sigma_data = 0.5,
        P_mean = -1.2,              
        P_std = 1.2,
        phase_aug_rate=0.0
    )


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



def read_from_config(config_path):
    with open(config_path, 'r') as config_file:
        yaml_config = yaml.safe_load(config_file)["Experiment"]
    config_data = {}
    
    config_data["Experiment_name"] = yaml_config["Experiment_name"]
    config_data["Project_name"] = yaml_config["Project_name"]
    
    config_data.update(yaml_config["model_params"].items())
    config_data.update(yaml_config["dataset_params"].items())
    config_data.update(yaml_config["data_augmentation"].items())
    config_data.update(yaml_config["checkpointing_params"].items())
    config_data.update(yaml_config["training_params"].items())
    config_data.update(yaml_config["loss_params"].items())
    config_data.update(yaml_config["cfg_params"].items())
    config_data.update(yaml_config["timestep_params"].items())
    config_data.update(yaml_config["diffusion_prameterization"].items())
    config_data.update(yaml_config["pf_ode_params"].items())
    config_data.update(yaml_config["ema_params"].items())
    config_data.update(yaml_config["edm_params"].items())
    config_data.update(yaml_config["optimizer_params"].items())
    config_data.update(yaml_config["Other_parms"].items())

    config_args = []
    for arg_name, arg_value in config_data.items():
        if isinstance(arg_value, bool):
            if arg_value==True:
                config_args.append("--" + arg_name)
            else:
                continue
        elif isinstance(arg_value, list):
            config_args.append('--' + arg_name + '=' + str(arg_value))
        else:
            config_args.append('--' + arg_name + '=' + str(arg_value))
        
    return config_args, yaml_config


def create_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--seed", type=int, default=5031,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--train_file", type=str, default="data/train_audiocaps.json",
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--num_examples", type=int, default=-1,
        help="How many audio samples to use for training and validation from entire training dataset.",
    )
    parser.add_argument(
        "--text_encoder_name", type=str, default="google/flan-t5-large",
        help="Text encoder identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unet_model_config", type=str, default="configs/diffusion_model_config.json",
        help="UNet model config json path.",
    )
    parser.add_argument(
        "--ctm_unet_model_config", type=str, default="configs/diffusion_model_config.json",
        help="CTM's UNet model config json path.",
    )
    parser.add_argument(
        "--freeze_text_encoder", action="store_true", default=False,
        help="Freeze the text encoder model.",
    )
    parser.add_argument(
        "--text_column", type=str, default="captions",
        help="The name of the column in the datasets containing the input texts.",
    )
    parser.add_argument(
        "--audio_column", type=str, default="location",
        help="The name of the column in the datasets containing the audio paths.",
    )
    parser.add_argument(
        "--tango_data_augment", action="store_true", default=False,
        help="Augment training data by tango's data augmentation.",
    )
    parser.add_argument(
        "--augment_num", type=int, default=2,
        help="number of augment training data.",
    )
    parser.add_argument(
        "--uncond_prob", type=float, default=0.1,
        help="Dropout rate of conditon text.",
    )
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Add prefix in text prompts.",
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=6,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=40,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="",
        help="Where to store the final model."
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="input audio duration"
    )
    parser.add_argument(
        "--checkpointing_steps", type=str, default="best",
        help="Whether the various states should be saved at the end of every 'epoch' or 'best' whenever validation loss decreases.",
    )

    parser.add_argument(
        "--model_grad_clip_value", type=float, default=1000.,
        help="Clipping value for gradient of model"
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default="",
        help="If the training should continue from a local checkpoint folder.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='bf16',
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--with_tracking", action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    
    # Additional params added
    parser.add_argument(
        "--teacher_model_path", type=str, default="ckpt/teacher/pytorch_model_2_sigma_025.bin",
        help="Path to teacher model ckpt",
    )
    parser.add_argument(
        "--stage1_path", type=str, default="ckpt/audioldm-s-full.ckpt",
        help="Path to stage1 model ckpt",
    )
    parser.add_argument(
        "--text_audio_pair_dataset", action="store_true",
        help="True if the dataset is text to audio pair and False for only audio dataset"
    )
    parser.add_argument(
        "--load_mean_std_state_path", type=str, default="",
        help="Load the mean and std from previously saved state path",
    )    
    parser.add_argument(
        "--Experiment_name", type=str, default="Experiment_1",
        help="Name of the experiment for WandB"
    )
    parser.add_argument(
        "--Project_name", type=str, default="Latent Sound EDM Evaluation",
        help="project name for wnadb"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=1,
        help="Test dataset batch size"
    )
    parser.add_argument(
        "--test_file", type=str, default="",
        help="Test dataset CSV path"
    )
    parser.add_argument(
        "--eval_num_steps", type=int, default=1,
        help="Number of steps for evaluation"
    )
    parser.add_argument(
        "--nu", type=float, default=1.0,
        help="Guidance scale"
    )
    parser.add_argument(
        "--omega", type=float, default=1.0,
        help="omega"
    )
    parser.add_argument(
        "--sampler", type=str, default="deterministic",
        help="omega"
    )
    parser.add_argument(
        "--sampling_gamma", type=float, default=1.0,
        help="sampling sigma"
    )
    parser.add_argument(
        "--reference_dir", type=str, default="",
        help="reference dir for sigma"
    )
    parser.add_argument(
        "--diffusion_model_type", type=str, default="unet",
        help="Type of diffusion model to use UNET/DIT",
    )
    parser.add_argument(
        "--dit_model_config", type=str, default="",
        help="DIT model config json path.",
    )
    parser.add_argument(
        "--lr_scheduler_type", type=SchedulerType, default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999,
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-08,
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--test_file_ACID", type=str,
        help=""
    )
    parser.add_argument(
        "--version", type=str, default="v2",
        help="Selects the version of SoundCTM DIT 1B v1 or v2, default is v2"
    )
    parser.add_argument(
        "--clap_text_branch_projection", action="store_true",
        help="If set, the output from the text branch will be passed through a projection layer before being used for diffusion training. Defaults to True if version is set to v2"
    )
    parser.add_argument(
        "--evaluation_interval", type=int, default=5000,
        help="Step interval for evaluation loop"
    )
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--config', type=str, help='Path to the YAML configuration file')
    
    defaults = dict()
    defaults.update(ctm_train_defaults())
    defaults.update(ldm_defaults())
    defaults.update()
    
    add_dict_to_argparser(parser, defaults)

    
    args, remaining_args = parser.parse_known_args()
    if args.config:
        config_path = os.path.abspath(args.config)
        if os.path.isfile(config_path):
            config_args, yaml_config = read_from_config(args.config)
            args = parser.parse_args(config_args + remaining_args)
        else:
            raise Exception(f'Configuration file "{config_path}" not found......')
    else:
        args = parser.parse_args()

    # sanity checks
    if args.train_file is None:
        raise ValueError("Need a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."

    return args, yaml_config
