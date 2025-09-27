import argparse
import yaml
import os

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
        config_data = yaml.safe_load(config_file)["Experiment"]

    config_args = []
    for arg_name, arg_value in config_data.items():
        if isinstance(arg_value, bool):
            if arg_value==True:
                print(arg_name)
                config_args.append("--" + arg_name)
            else:
                continue
        elif isinstance(arg_value, list):
            print(arg_value, '--' + arg_name + '=' + str(arg_value))
            config_args.append('--' + arg_name + '=' + str(arg_value))
        else:
            config_args.append('--' + arg_name + '=' + str(arg_value))
        
    return config_args


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for text to audio generation task.")
    parser.add_argument(
        "--training_args", type=str, default="",
        help="Path for 'summary.jsonl' file saved during training."
    )
    parser.add_argument(
        "--output_dir", type=str, default="",
        help="Where to store the output."
    )
    parser.add_argument(
        "--seed", type=int, default=5031,
        help="A seed for reproducible inference."
    )
    parser.add_argument(
        "--text_encoder_name", type=str, default="",
        help="Text encoder identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--ctm_unet_model_config", type=str, default="",
        help="UNet model config json path.",
    )
    parser.add_argument(
        "--sampling_rate", type=float, default=44100,
        help="Sampling rate of training data",
    )
    parser.add_argument(
        "--target_length", type=float, default=10,
        help="Audio length of training data",
    )
    parser.add_argument(
        "--teacher_model", type=str, default="",
        help="Teacher model path"
    )
    parser.add_argument(
        "--ema_model", type=str, default="",
        help="student model path -- ema ckpt"
    )
    parser.add_argument(
        "--sampler", type=str, default='determinisitc',
        help="Inference sampling methods. You can choose ['determinisitc' (gamma=0), 'cm_multistep' (gamma=1), 'gamma_multistep']."
    )
    parser.add_argument(
        "--sampling_gamma", type=float, default=0.9,
        help="\gamma for gamma-sampling if we use 'gamma_multistep'."
    )
    parser.add_argument(
        "--test_file", type=str, default="",
        help="json file containing the test prompts for generation."
    )
    parser.add_argument(
        "--test_references", type=str, default="",
        help="Folder containing the test reference wav files."
    )
    parser.add_argument(
        "--num_steps", type=int, default=1,
        help="How many denoising steps for generation.",
    )
    parser.add_argument(
        "--nu", type=float, default=1.,
        help="Guidance scale for \nu interpolation."
    )
    parser.add_argument(
        "--omega", type=float, default=3.5,
        help="Omega for student model."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="How many samples per prompt.",
    )
    parser.add_argument(
        "--sigma_data", type=float, default=0.25,
        help="Sigma data",
    )
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Add prefix in text prompts.",
    )
    parser.add_argument(
        "--stage1_path", type=str, default="",
        help="Path to stage1 model ckpt",
    )
    parser.add_argument(
        "--diffusion_model_type", type=str, default="",
        help="Type of diffusion model to use DIT",
    )
    parser.add_argument(
        "--dit_model_config", type=str, default="",
        help="DIT model config json path.",
    )
    parser.add_argument(
        "--Experiment_name", type=str, default="Experiment_1",
        help="Name of the experiment for WandB"
    )
    parser.add_argument(
        "--Project_name", type=str, default="",
        help="project name for wnadb"
    )
    parser.add_argument(
        "--sigma_max", type=float, default=80.0,
        help="Sigma data",
    )
    parser.add_argument(
        "--sigma_min", type=float, default=0.002,
        help="Sigma data",
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
        "--clap_model_path", type=str, default="",
        help="Path to clap model ckpt",
    )
    parser.add_argument(
        "--version", type=str, default="v2",
        help="Selects the version of SoundCTM DIT 1B v1 or v2, default is v2"
    )
    parser.add_argument(
        "--clap_text_branch_projection", action="store_false",
        help="If set, the output from the text branch will be passed through a projection layer before being used for diffusion training. Defaults to True if version is set to v2"
    )
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--config', type=str, help='Path to the YAML configuration file')
    
    args, remaining_args = parser.parse_known_args()
    if args.config:
        config_path = os.path.abspath(args.config)
        if os.path.isfile(config_path):
            config_args = read_from_config(args.config)
            args = parser.parse_args(config_args + remaining_args)
        else:
            raise Exception(f'Configuration file "{config_path}" not found......')
    else:
        args = parser.parse_args()
    
    return args 