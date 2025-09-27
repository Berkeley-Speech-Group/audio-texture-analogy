"""This script used for knowledge distillation from teacher to student"""

import warnings
import json
import logging
import math
import os
import random
import time
import datasets
import diffusers
import numpy as np
import pandas as pd
import torch as th
import transformers
import wandb
import torchaudio
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from ctm.script_util import (
    create_ema_and_scales_fn,
    create_model_and_diffusion,
)
from pedalboard.io import AudioFile
from torchaudio import transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from ctm.train_util import CTMTrainLoop
from sa_edm.models_edm import build_stage1_models
from sa_edm.training_args_config import create_argparser
from sa_edm.util import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T, calculate_mean_std



os.environ['WANDB_API_KEY'] = ""
# os.environ['WANDB_INIT_TIMEOUT'] = "180"

warnings.filterwarnings("ignore", category=UserWarning)
logger = get_logger(__name__)

# # USE THE BELOW CODE ONLY IF you are getting nccl timeout error in distributed environment
# import torch.distributed as dist
# TIMEOUT_MINUTES=120
# dist.init_process_group(
#     backend="nccl",
#     timeout=timedelta(minutes=TIMEOUT_MINUTES)
# )
# # Set NCCL debug and timeout settings
# os.environ["NCCL_DEBUG"] = "INFO"  # Optional: Provides more logging information for debugging
# os.environ["NCCL_BLOCKING_WAIT"] = "1"  # Forces NCCL to block on errors instead of failing silently
# os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # Optional: Helps with asynchronous error management
# os.environ["NCCL_TIMEOUT"] = "7200"


def rand_fix(seed):
    """
    Fixes the random seed for reproducibility across various libraries.
    Parameters:
    seed (int): The seed value to be used for random number generation.
    """

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True


class Text2AudioDataset(Dataset):
    """Dataset class for loading, preprocessing, combining samples into batch"""
    def __init__(
        self,
        dataset,
        prefix,
        text_column,
        audio_column,
        uncond_prob=0.0,
        num_examples=-1,
        text_encoder='t5',
        sample_size=441000,
        sample_rate=44100,
        phase_aug_rate=0.0,
        random_crop=True,
        force_channels="mono",
        text_audio_pair_dataset=False
        ):
        """
        Initializes the training dataset for the CTM model.
        Parameters:
        - dataset (dict): The dataset containing audio and text data.
        - prefix (str): The prefix to be added to each text input.
        - text_column (str): The column name for text captions in the dataset.
        - audio_column (str): The column name for audio file path in the dataset.
        - uncond_prob (float, optional): The probability for unconditional training, part of CFG training. Default is 0.0.
        - num_examples (int, optional): The number of examples to use from the dataset. Default is -1 (use all examples).
        - text_encoder (str, optional): The type of text encoder to use ('t5' or 'clap'). Default is 't5'.
        - sample_size (int, optional): The size of the audio samples. Default is 441000.
        - sample_rate (int, optional): The sample rate of the audio. Default is 44100.
        - phase_aug_rate (float, optional): The rate of phase augmentation. Default is 0.0.
        - random_crop (bool, optional): Whether to randomly crop the audio samples. Default is True.
        - force_channels (str, optional): The type of audio channels to force ('mono' or 'stereo'). Default is 'mono'.
        - text_audio_pair_dataset (bool, optional): Whether the dataset contains paired text and audio data OR only audio data. Specify True if paired. Default is False.
        """

        self.audios = list(dataset[audio_column])
        self.text_encoder = text_encoder

        self.uncond_prob = uncond_prob

        self.augs = th.nn.Sequential(
            PhaseFlipper(phase_aug_rate),
        )

        self.pad_crop = PadCrop_Normalized_T(
            sample_size, sample_rate, randomize=random_crop)

        self.force_channels = force_channels

        self.encoding = th.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else th.nn.Identity(),
            Mono() if self.force_channels == "mono" else th.nn.Identity(),
        )
        self.sr = sample_rate

        self.text_audio_pair_dataset = text_audio_pair_dataset

        self.mapper = {}
        if text_encoder == 'clap':
            if self.text_audio_pair_dataset:
                inputs = list(dataset[text_column])
                self.inputs = [prefix + inp for inp in inputs]
                self.indices = list(range(len(self.inputs)))
                for index, audio, text in zip(self.indices, self.audios, inputs):
                    self.mapper[index] = [audio, text]
                if num_examples != -1:
                    self.inputs, self.audios = self.inputs[:
                                                           num_examples], self.audios[:num_examples]
                    self.indices = self.indices[:num_examples]
            else:
                self.indices = list(range(len(self.audios)))
                for index, audio in zip(self.indices, self.audios):
                    self.mapper[index] = [audio]
                if num_examples != -1:
                    self.audios = self.audios[:num_examples]
                    self.indices = self.indices[:num_examples]

        else:
            inputs = list(dataset[text_column])
            self.inputs = [prefix + inp for inp in inputs]
            self.indices = list(range(len(self.inputs)))
            for index, audio, text in zip(self.indices, self.audios, inputs):
                self.mapper[index] = [audio, text]
            if num_examples != -1:
                self.inputs, self.audios = self.inputs[:num_examples], self.audios[:num_examples]
                self.indices = self.indices[:num_examples]

    def __len__(self):
        """
        Returns the length of the dataset based on the text encoder type.
        Returns:
            int: The length of the dataset.
        """

        if self.text_encoder == 'clap':
            return len(self.audios)
        else:
            return len(self.inputs)

    def get_num_instances(self):
        """
        Returns the number of instances based on the text encoder type.
        Returns:
            int: The number of instances.
        """

        if self.text_encoder == 'clap':
            return len(self.audios)
        else:
            return len(self.inputs)

    def load_file(self, filename):
        """
        Load an audio file and resample it if necessary.
        Parameters:
        filename (str): The path to the audio file to be loaded. The file can be in mp3 format or any format supported by torchaudio.
        Returns:
        torch.Tensor: The loaded audio data as a tensor.
        """

        ext = filename.split(".")[-1]
        if ext == "mp3":
            with AudioFile(filename) as f:
                audio = f.read(f.frames)
                audio = th.from_numpy(audio)
                in_sr = f.samplerate
        else:
            audio, in_sr = torchaudio.load(filename, format=ext)

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr)
            audio = resample_tf(audio)

        return audio

    def __getitem__(self, index):
        """
        Retrieve the audio and corresponding information at the specified index.
        Parameters:
            index (int): The index of the audio file and corresponding data to retrieve.
        Returns:
            tuple: A tuple containing the following elements:
                - text (str): The text associated with the audio, or an empty string based on a random condition.
                - audio (Tensor): The processed audio tensor.
                - index (int): The index of the audio file.
                - emb_audio (Tensor, optional): The resampled audio tensor for CLAP encoding, if applicable.
        """

        audio_filename = self.audios[index]
        start_time = time.time()
        audio = self.load_file(audio_filename)
        audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(
            audio)

        # Run augmentations on this sample (including random crop)
        if self.augs is not None:
            audio = self.augs(audio)

        audio = audio.clamp(-1, 1)

        # Encode the file to assist in prediction
        if self.encoding is not None:
            audio = self.encoding(audio)
        info = {}
        info["path"] = audio_filename
        info["timestamps"] = (t_start, t_end)
        info["seconds_start"] = seconds_start
        info["seconds_total"] = seconds_total
        info["padding_mask"] = padding_mask
        end_time = time.time()
        info["load_time"] = end_time - start_time

        if self.text_encoder == 'clap':
            if self.text_audio_pair_dataset:
                text = self.inputs[index]
                text = "" if random.random() < self.uncond_prob else text
                return text, audio, self.indices[index]
            else:
                audio_tmp = th.zeros_like(
                    audio) if random.random() < self.uncond_prob else audio
                resample_tf = T.Resample(self.sr, 48000)
                emb_audio = resample_tf(audio_tmp)  # CLAP requires 48 kHz.
                return emb_audio, audio, self.indices[index]
        else:
            text = self.inputs[index]
            text = "" if random.random() < self.uncond_prob else text
            return text, audio, self.indices[index]

    def collate_fn(self, data):
        """Collate samples into batch"""
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]


def main():
    """Main function for starting execution of this script"""

    args, yaml_config = create_argparser()

    # # Enable deepspeed plugin for memory efficient training
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_clipping=100.0)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # mixed_precision=args.mixed_precision,
        # deepspeed_plugin=deepspeed_plugin,
        # kwargs_handlers=[ddp_kwargs],
        # **accelerator_log_kwargs
    )

    if accelerator.is_main_process:
        logger.info("-----------------------------------------------------------------------------------------------------------------------------")
        logger.info("Training args: ", vars(args))
        logger.info("-----------------------------------------------------------------------------------------------------------------------------")
        logger.info("Created Accelerator")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        datasets.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        rand_fix(args.seed)

    # Handle output directory creation and wandb tracking
    if accelerator.is_main_process:
        if args.output_dir is None or args.output_dir == "":
            args.output_dir = "saved/" + str(int(time.time()))

            if not os.path.exists("saved"):
                os.makedirs("saved")

            os.makedirs(args.output_dir, exist_ok=True)

        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        os.makedirs(f"{args.output_dir}/outputs/", exist_ok=True)
        with open(f"{args.output_dir}/summary.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(vars(args))) + "\n\n")

        accelerator.project_configuration.automatic_checkpoint_naming = False

        wandb.init(project=args.Project_name, name=args.Experiment_name,
                   config=yaml_config)  # , resume="must", id="u85gxd48")

    accelerator.wait_for_everyone()
    if args.with_tracking:
        accelerator.init_trackers("text_to_audio_diffusion")

    # Get the datasets
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file

    extension = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    text_column, audio_column = args.text_column, args.audio_column

    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )

    logger.info("Loading stage1(VAE) model")
    vae = build_stage1_models(ckpt_folder_path=args.stage1_path)
    vae.requires_grad_(False)
    vae.eval()
    vae.to(accelerator.device)
    logger.info("VAE has been loaded")

    # Load Model
    logger.info("creating the student model")
    model, diffusion = create_model_and_diffusion(args)
    model.train()
    logger.info("Done....")

    # Load teacher model
    if len(args.teacher_model_path) > 0:  # path to the teacher score model.
        logger.info(
            f"loading the teacher model from {args.teacher_model_path}")
        teacher_model, _ = create_model_and_diffusion(args, teacher=True)

        if os.path.exists(args.teacher_model_path):
            logger.info(
                f"Loading the teacher model from path {args.teacher_model_path}................")
            model_ckpt = th.load(args.teacher_model_path,
                                 map_location=accelerator.device)
            # teacher_model.load_state_dict(model_ckpt["module"])

            new_state_dict = {}
            for key, value in model_ckpt.items():
                new_key = key.replace("module.", "")  # Add "module." prefix
                new_state_dict[new_key] = value

            teacher_model.load_state_dict(new_state_dict)

            logger.info("Done.............")
        teacher_model.eval()
        logger.info("Now copying the teacher model params to student....")
        # Initialize model parameters with teacher model

        if teacher_model.model_type == "unet":
            logger.info(
                "Copying the UNET model parameters....................")
            copied_params = set()
            mismatched_params = []

            for dst_name, dst in model.unet.named_parameters():
                matched = False
                for src_name, src in teacher_model.unet.named_parameters():
                    if dst_name == src_name or dst_name == '.'.join(src_name.split('.')[1:]):
                        if dst.data.shape == src.data.shape:
                            dst.data.copy_(src.data)
                            copied_params.add(dst_name)
                            matched = True
                            logger.info(f"Copied parameter: {dst_name}")
                            break
                        else:
                            mismatched_params.append((dst_name, src_name))

                if not matched:
                    logger.warning(f"No match found for parameter: {dst_name}")

            total_params = sum(1 for _ in model.unet.named_parameters())
            logger.info(
                f"Copied {len(copied_params)} out of {total_params} parameters")

            if mismatched_params:
                logger.warning("Mismatched parameters (different shapes):")
                for dst, src in mismatched_params:
                    logger.warning(f"  {dst} (student) vs {src} (teacher)")

            if len(copied_params) < total_params:
                logger.warning(
                    "Some parameters were not copied. Check model structures.")

        elif teacher_model.model_type == "dit":
            logger.info(
                "Copying the DIT model parameters.....................")
            for dst_name, dst in model.dit.named_parameters():
                for src_name, src in teacher_model.dit.named_parameters():
                    if dst_name in ['.'.join(src_name.split('.')[1:]), src_name]:
                        dst.data.copy_(src.data)
                        break
        else:
            raise NotImplementedError

        for dst_name, dst in model.text_encoder.named_parameters():
            for src_name, src in teacher_model.text_encoder.named_parameters():
                if dst_name in ['.'.join(src_name.split('.')[1:]), src_name]:
                    dst.data.copy_(src.data)
                    break
        teacher_model.requires_grad_(False)
        teacher_model.eval()
        logger.info(
            f"Initialized parameters of student (online) model synced with the \
                teacher model from {args.teacher_model_path}")

    else:
        teacher_model = None
    # Load the target model for distillation, if path specified.
    logger.info("creating the target model")
    target_model, _ = create_model_and_diffusion(args)
    logger.info("Copy parameters of student model with the target_model model")
    for dst, src in zip(target_model.parameters(), model.parameters()):
        dst.data.copy_(src.data)

    target_model.requires_grad_(False)
    target_model.train()

    logger.info("Done....")

    target_model.to(accelerator.device)
    teacher_model.to(accelerator.device)

    # Define dataloader
    logger.info("creating data loader...")
    if args.prefix:
        prefix = args.prefix
    else:
        prefix = ""

    with accelerator.main_process_first():
        train_dataset = Text2AudioDataset(
            raw_datasets["train"],
            prefix,
            text_column,
            audio_column,
            args.uncond_prob,
            args.num_examples,
            args.text_encoder_name,
            sample_size=int(args.sampling_rate * args.target_sec),
            sample_rate=args.sampling_rate,
            phase_aug_rate=args.phase_aug_rate,
            random_crop=args.phase_aug_rate,
            force_channels=args.audio_channel,
            text_audio_pair_dataset=args.text_audio_pair_dataset
        )
        num_params = train_dataset.get_num_instances()
        logger.info(f"Num instances in train: {num_params}")

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size, collate_fn=train_dataset.collate_fn)

    if args.load_mean_std_state_path == "":
        logger.info("Calculating mean and std for whole dataset from scratch")
        z_mean, z_std = calculate_mean_std(
            train_dataset, vae, accelerator.device)
    else:
        logger.info(
            f"Loading the previously calculated mean and std from {args.load_mean_std_state_path}")
        progress_state = th.load(os.path.join(args.load_mean_std_state_path))
        z_mean = progress_state['z_mean'].to(accelerator.device)
        z_std = progress_state['z_std'].to(accelerator.device)

    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    # Optimizer
    if args.freeze_text_encoder:
        logger.info("Freezing the text encoder.............")
        for param in model.text_encoder.parameters():
            param.requires_grad = False
            model.text_encoder.eval()

    if model.model_type == "unet":
        optimizer_parameters = model.unet.parameters()
        logger.info("Optimizing CTM UNet parameters.")
    elif model.model_type == "dit":
        optimizer_parameters = model.dit.parameters()
        logger.info("Optimizing CTM DIT parameters.")
    else:
        raise NotImplementedError

    num_trainable_parameters = sum(p.numel()
                                   for p in model.parameters() if p.requires_grad)
    logger.info(f"Num CTM model trainable parameters: {num_trainable_parameters}")

    optimizer = th.optim.RAdam(
        optimizer_parameters, lr=args.lr,
        weight_decay=args.weight_decay,
    )

    overrode_total_training_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if args.total_training_steps is None:
        args.total_training_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_total_training_steps = True

    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    model, optimizer, train_dataloader, = accelerator.prepare(
        model, optimizer, train_dataloader)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if overrode_total_training_steps:
        args.total_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.total_training_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.

    # Train
    total_batch_size = (args.per_device_train_batch_size + args.augment_num) * \
        args.gradient_accumulation_steps * accelerator.num_processes
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device + augment_num =\
            {args.per_device_train_batch_size} + {args.augment_num}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation & data augmentation) = \
            {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.total_training_steps}")

    resume_epoch = 0
    resume_step = 0
    resume_global_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            progress_state = th.load(os.path.join(
                args.resume_from_checkpoint, "progress_state.pth"), map_location=accelerator.device)
            resume_step = progress_state['completed_steps']
            resume_global_step = progress_state['completed_global_steps']
            resume_epoch = progress_state['completed_epochs']
            accelerator.load_state(args.resume_from_checkpoint)

            state_dict = th.load(os.path.join(args.resume_from_checkpoint,
                                 f"target_{resume_step:06d}.pt"), map_location=accelerator.device)
            target_model.load_state_dict(state_dict)
            target_model.requires_grad_(False)
            target_model.train()
            target_model.to(accelerator.device)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)

    CTMTrainLoop(
        model=model,
        target_model=target_model,
        teacher_model=teacher_model,
        latent_decoder=vae,
        z_mean=z_mean,
        z_std=z_std,
        ema_scale_fn=ema_scale_fn,
        diffusion=diffusion,
        data=train_dataloader,
        args=args,
        accelerator=accelerator,
        opt=optimizer,
        resume_step=resume_step,
        resume_global_step=resume_global_step,
        resume_epoch=resume_epoch,
    ).run_loop()


if __name__ == "__main__":
    main()
