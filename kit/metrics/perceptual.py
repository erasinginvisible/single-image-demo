import torch
from PIL import Image
from torchvision import transforms
from .lpips import LPIPS


# Normalize image tensors
def normalize_tensor(images, norm_type):
    assert norm_type in ["imagenet", "naive"]
    # Two possible normalization conventions
    if norm_type == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean, std)
    elif norm_type == "naive":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize = transforms.Normalize(mean, std)
    else:
        assert False
    return torch.stack([normalize(image) for image in images])


def to_tensor(images, norm_type="naive"):
    assert isinstance(images, list) and all(
        [isinstance(image, Image.Image) for image in images]
    )
    images = torch.stack([transforms.ToTensor()(image) for image in images])
    if norm_type is not None:
        images = normalize_tensor(images, norm_type)
    return images


def load_perceptual_models(metric_name, mode, device=torch.device("cuda")):
    assert metric_name in ["lpips"]
    if metric_name == "lpips":
        assert mode in ["vgg", "alex"]
        perceptual_model = LPIPS(net=mode).to(device)
    else:
        assert False
    return perceptual_model


# Compute metric between two images
def compute_metric(image1, image2, perceptual_model, device=torch.device("cuda")):
    assert isinstance(image1, Image.Image) and isinstance(image2, Image.Image)
    image1_tensor = to_tensor([image1]).to(device)
    image2_tensor = to_tensor([image2]).to(device)
    return perceptual_model(image1_tensor, image2_tensor).cpu().item()


# Compute LPIPS distance between two images
def compute_lpips(image1, image2, mode="alex", device=torch.device("cuda")):
    perceptual_model = load_perceptual_models("lpips", mode, device)
    return compute_metric(image1, image2, perceptual_model, device)


# Compute metrics between pairs of images
def compute_perceptual_metric_repeated(
    images1,
    images2,
    metric_name,
    mode,
    model,
    device,
):
    # Accept list of PIL images
    assert isinstance(images1, list) and isinstance(images1[0], Image.Image)
    assert isinstance(images2, list) and isinstance(images2[0], Image.Image)
    assert len(images1) == len(images2)
    if model is None:
        model = load_perceptual_models(metric_name, mode).to(device)
    return (
        model(to_tensor(images1).to(device), to_tensor(images2).to(device))
        .detach()
        .cpu()
        .numpy()
        .flatten()
        .tolist()
    )


# Compute LPIPS distance between pairs of images
def compute_lpips_repeated(
    images1,
    images2,
    mode="alex",
    model=None,
    device=torch.device("cuda"),
):
    return compute_perceptual_metric_repeated(
        images1, images2, "lpips", mode, model, device
    )
