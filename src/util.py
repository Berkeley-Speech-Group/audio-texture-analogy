from torchvision.transforms import ToPILImage, Normalize
import torch

def read_txt(file_name):
    # read txt file and remove \n
    with open(file_name, 'r') as f:
        lines = f.readlines()
    lines = [line.rstrip() for line in lines]
    return lines

def img2jpg(normalized_image_tensor, normalize_transform=Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))):
    """
    Denormalizes a normalized image tensor using the provided Normalize transform, and converts it to a PIL image.
    
    Args:
        normalized_image_tensor (torch.Tensor): Normalized image tensor.
        normalize_transform (Normalize): Normalize transform used for normalization.
        
    Returns:
        torch.Tensor: PIL image.
    """
    denormalize_transform = Normalize(
        mean=[-m/s for m, s in zip(normalize_transform.mean, normalize_transform.std)],
        std=[1/s for s in normalize_transform.std]
    )
    to_pil = ToPILImage()
    image_tensor = to_pil(denormalize_transform(normalized_image_tensor))
    return image_tensor

def load_clap_model(clap_model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    clap_model.load_state_dict(checkpoint['clap_state_dict'])
    # clap_model.load_state_dict(checkpoint['clap_model_state_dict'])


    return clap_model
