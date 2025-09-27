"""This script evaluates the generated samples against the ground truth samples"""
import os
import argparse
import sys
import torch
import pandas as pd
import laion_clap
import librosa
import torch
import pyloudnorm as pyln
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm
from stable_audio_metrics.src.openl3_fd import openl3_fd
from stable_audio_metrics.src.passt_kld import passt_kld
from stable_audio_metrics.src.clap_score import clap_score
from stable_audio_metrics.src.openl3_fd import calculate_embd_statistics, calculate_frechet_distance
from clap_module.factory import load_state_dict


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

_SIGMA = 10
_SCALE = 1000


def int16_to_float32(x):
    """
    Convert a NumPy array of int16 values to float32.
    Parameters:
        x (numpy.ndarray): A NumPy array of int16 values.
    Returns:
        numpy.ndarray: A NumPy array of float32 values, scaled from the int16 input.
    """

    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    """
    Converts a NumPy array of float32 values to int16 values.
    This function clips the input array values to the range [-1.0, 1.0] and then scales them to the range of int16 
    (-32768 to 32767).
    Parameters:
        x (numpy.ndarray): A NumPy array of float32 values to be converted.
    Returns:
        numpy.ndarray: A NumPy array of int16 values.
    """

    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def parse_args_cmd():
    """
    Parse command-line arguments for the text to audio generation task.
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    Arguments:
        --reference_dir (str): Folder containing the test reference wav files. Default is "/data/audiocaps/test/audio".
        --reference_path_acid (str): Folder containing the generated wav files. Default is "/path/to/generated_dir".
        --generated_dir (str): Folder containing the generated wav files. Default is "/path/to/generated_dir".
        --reference_csv (str): Path to the reference CSV file. Default is "/path/to/ref_csv".
        --clap_model_path (str): Path to the CLAP model checkpoint. Default is "630k-audioset-best.pt".
    """
    parser = argparse.ArgumentParser(description="Inference for text to audio generation task.")
    parser.add_argument(
        "--reference_dir", type=str, default="audiocaps/test/audio/",
        help="Folder containing the test reference wav files."
    )
    
    parser.add_argument(
        "--reference_path_acid", type=str, default="data/clap_test.csv",
        help="Folder containing the generated wav files."
    )

    parser.add_argument(
        "--generated_dir", type=str, default="output/generated_dir",
        help="Folder containing the generated wav files."
    )
    parser.add_argument(
        "--reference_csv", type=str, default="data/test.csv",
        help="path for csv"
    )
    parser.add_argument(
        "--clap_model_path", type=str, default="checkpoint/630k-audioset-best.pt",
        help="path for csv"
    )
    args = parser.parse_args()
    return args

def mmd(x, y):
    """Memory-efficient MMD implementation in JAX.

    This implements the minimum-variance/biased version of the estimator described
    in Eq.(5) of
    https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    As described in Lemma 6's proof in that paper, the unbiased estimate and the
    minimum-variance estimate for MMD are almost identical.

    Note that the first invocation of this function will be considerably slow due
    to JAX JIT compilation.

    Args:
      x: The first set of embeddings of shape (n, embedding_dim).
      y: The second set of embeddings of shape (n, embedding_dim).

    Returns:
      The MMD distance between x and y embedding sets.
    """
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0)))
    )
    k_xy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )
    k_yy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )

    return _SCALE * (k_xx + k_yy - 2 * k_xy)

def print_metrics(metrics, dirs=None):
    """
    Prints the evaluation metrics and directories in a formatted table.
    Args:
        metrics (list of tuples): A list of tuples where each tuple contains the metric name and its value.
        dirs (list of tuples, optional): A list of tuples where each tuple contains the directory name and its path. Defaults to an empty list.
    Returns:
        None
    """

    dir_table = PrettyTable()
    dir_table.field_names = ["Dir", "Path"]
    dir_table.max_width["Path"] = 70
    dir_table.add_rows(dirs)

    metrics_table = PrettyTable()
    metrics_table.field_names = ["Metric name", "Metric value"]
    metrics_table.add_rows(metrics)
    
    print("\n\n")
    print("-------------------------------Evaluation Result---------------------------------")
    print("\nDirectories:")
    print(dir_table)
    print("\nMetrics:")
    print(metrics_table)
    

def evaluate_generated(args, device="cuda", clap_model_path="630k-audioset-best.pt"):
    """
    Evaluates generated audio files against reference audio files using various metrics.
    
    Parameters:
    args (dict): A dictionary containing the following keys:
        - "generated_dir" (str): Path to the directory containing generated audio files.
        - "reference_dir" (str): Path to the directory containing reference audio files.
        - "reference_path_acid" (str): Path to the CSV file containing reference captions.
    device (str, optional): The device to run the CLAP model on. Default is "cuda".
    clap_model_path (str, optional): Path to the pre-trained CLAP model. Default is "630k-audioset-best.pt".
    
    Returns:
    list: A list of lists where each sublist contains a metric name and its corresponding value.
    
    Metrics:
    - FD OpenL3: Frechet Distance using OpenL3 embeddings.
    - CLAP score: Score calculated using the CLAP model.
    - KL Passt score: Kullback-Leibler divergence score using Passt model.
    - FD CLAP: Frechet Distance using CLAP embeddings.
    - MMD CLAP score: Maximum Mean Discrepancy using CLAP embeddings.
    """


    # ---------------------------------------------------------------------------------
    # FD OpenL3
    # ---------------------------------------------------------------------------------
    # FD defaults
    model_channels = 1
    model_sr = 44100
    type = 'env'
    hop = 0.5
    batch = 1
    # Calculation of FD OpenL3
    fd_openl3 = openl3_fd(
                    channels=model_channels, samplingrate=model_sr, content_type=type,
                    openl3_hop_size=hop, eval_path=args["generated_dir"], ref_path=args["reference_dir"], 
                    batching=batch
                    )
    
    #---------------------------------------------------------------------------------
    # CLAP score
    #---------------------------------------------------------------------------------
    df = pd.read_csv(args["reference_path_acid"])
    id2text = df.set_index('audiocap_id')['caption'].to_dict()
    clp_score = clap_score(id2text, 
                        args["generated_dir"], 
                        audio_files_extension='.wav', 
                        clap_model="630k-audioset-best.pt", 
                        clap_model_path=clap_model_path)

    #---------------------------------------------------------------------------------
    # KL Passt
    #---------------------------------------------------------------------------------
    df = pd.read_csv(args["reference_path_acid"])
    id2text = df.set_index('audiocap_id')['caption'].to_dict()
    audiocaps_ids = df['audiocap_id'].tolist()
    kl_passt_score = passt_kld(ids=audiocaps_ids, 
                eval_path=args["generated_dir"], 
                ref_path=args["reference_dir"], 
                no_ids=[],
                collect='mean')
    # ---------------------------------------------------------------------------------
    # FD CLAP Score and MMD CLAP distance
    # ---------------------------------------------------------------------------------
    
    # Load the CLAP model
    model = laion_clap.CLAP_Module(enable_fusion=False, device=device, amodel="HTSAT-tiny")
    pkg = load_state_dict(clap_model_path)
    pkg.pop('text_branch.embeddings.position_ids', None)
    model.model.load_state_dict(pkg)
    model.eval()

    # Get the audio embeddings for both generated audios and ground truth audios
     
    # following documentation from https://github.com/LAION-AI/CLAP
    generated_embeddings = []
    for file_name in tqdm(os.listdir(args["generated_dir"])): 
        file_path = os.path.join(args["generated_dir"], file_name)
        with torch.no_grad():
            audio, _ = librosa.load(file_path, sr=48000, mono=True) # sample rate should be 48000
            audio = pyln.normalize.peak(audio, -1.0)
            audio = audio.reshape(1, -1) # unsqueeze (1,T)
            audio = torch.from_numpy(int16_to_float32(float32_to_int16(audio))).float()
            generated_embeddings.append(model.get_audio_embedding_from_data(x = audio, use_tensor=True).to("cpu").numpy())

    ground_truth_embeddings = []
    for file_name in tqdm(os.listdir(args["reference_dir"])): 
        file_path = os.path.join(args["reference_dir"], file_name)
        with torch.no_grad():
            audio, _ = librosa.load(file_path, sr=48000, mono=True) # sample rate should be 48000
            audio = pyln.normalize.peak(audio, -1.0)
            audio = audio.reshape(1, -1) # unsqueeze (1,T)
            audio = torch.from_numpy(int16_to_float32(float32_to_int16(audio))).float()
            ground_truth_embeddings.append(model.get_audio_embedding_from_data(x = audio, use_tensor=True).to("cpu").numpy())

    # Convert the embeddings to 2D numpy arrays shape: (n, emb_dim)
    generated_embeddings = np.array(generated_embeddings).squeeze(1)
    ground_truth_embeddings = np.array(ground_truth_embeddings).squeeze(1)

    # Calculate FD CLAP score using CLAP embeddings 
    gen_mu, gen_std = calculate_embd_statistics(generated_embeddings)
    gt_mu, gt_std = calculate_embd_statistics(ground_truth_embeddings)
    fd_clap = calculate_frechet_distance(gen_mu, gen_std, gt_mu, gt_std)
    
    # Calculate the MMD score using CLAP embeddings
    mmd_clap = mmd(generated_embeddings, ground_truth_embeddings).numpy().tolist()

    metrics = [
                ["FD OpenL3", fd_openl3],
                ["CLAP score", clp_score],
                ["KL Passt score", kl_passt_score],
                ["FD CLAP", fd_clap],
                ["MMD CLAP score", mmd_clap]
              ]

    return metrics

if __name__ == "__main__":
    args = vars(parse_args_cmd())
    metrics = evaluate_generated(args, clap_model_path=args["clap_model_path"])
    dirs = [
            ["Ground truth directory", args["reference_dir"]],
            ["Generated samples directory", args["generated_dir"]]
            ]
    print_metrics(metrics=metrics, dirs=dirs)
