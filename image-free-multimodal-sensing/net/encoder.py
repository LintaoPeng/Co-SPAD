import numpy as np
import scipy.io
import torch
from PIL import Image

from net.UDLSSPI1k_step2 import LSSPI_two


def get_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder(
    step1_path="./weights/UDLSSPI1k_step1.pth",
    encoder_path="./weights/encoder.pth",
    device=None,
):
    device = get_device(device)
    model = LSSPI_two(path=step1_path, map_location=device).to(device)
    state_dict = torch.load(encoder_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def _tensor_to_pil(image_tensor):
    image_tensor = image_tensor.detach().float().cpu()
    image_tensor = image_tensor - image_tensor.min()
    denom = image_tensor.max().clamp_min(1e-8)
    image_tensor = (image_tensor / denom).clamp(0.0, 1.0)
    array = image_tensor.mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def reconstruct_images(
    model,
    feature_path="./features/features.mat",
    device=None,
    image_index=None,
):
    device = get_device(device)

    data = scipy.io.loadmat(feature_path)
    if "data" not in data:
        raise KeyError("The feature .mat file must contain a variable named 'data'.")

    features = data["data"]
    indices = range(len(features)) if image_index is None else [image_index]
    results = []

    with torch.no_grad():
        for index in indices:
            feature = torch.from_numpy(features[index].astype(np.float32)).unsqueeze(0).to(device)
            output = model(feature)[0][0]

            results.append(
                {
                    "index": index,
                    "image": _tensor_to_pil(output),
                }
            )

    return results
