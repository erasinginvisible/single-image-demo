import os
import io
import numpy as np
import onnxruntime as ort
from PIL import Image
import dotenv

dotenv.load_dotenv()

GT_MESSAGE = os.environ["GT_MESSAGE"]


QUALITY_COEFFICIENTS = {
    "psnr": -0.0022186489180419534,
    "ssim": -0.11337077856710862,
    "nmi": -0.09878221979274945,
    "lpips": 0.3412626374646173,
}

QUALITY_OFFSETS = {
    "psnr": 43.54757854447622,
    "ssim": 0.984229018845295,
    "nmi": 1.7536553655336136,
    "lpips": 0.014247652621287854,
}


def compute_performance(image):
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.log_severity_level = 3
    model = ort.InferenceSession(
        "./kit/models/stable_signature.onnx",
        sess_options=session_options,
    )
    inputs = np.stack(
        [
            (
                (
                    np.array(
                        image,
                        dtype=np.float32,
                    )
                    / 255.0
                    - [0.485, 0.456, 0.406]
                )
                / [0.229, 0.224, 0.225]
            )
            .transpose((2, 0, 1))
            .astype(np.float32)
        ],
        axis=0,
    )

    outputs = model.run(
        None,
        {
            "image": inputs,
        },
    )
    decoded = (outputs[0] > 0).astype(int)[0]
    gt_message = np.array([int(bit) for bit in GT_MESSAGE])
    return 1 - np.mean(gt_message != decoded)


from .metrics import (
    compute_image_distance_repeated,
    load_perceptual_models,
    compute_perceptual_metric_repeated,
    load_aesthetics_and_artifacts_models,
    compute_aesthetics_and_artifacts_scores,
)


def compute_quality(attacked_image, clean_image, quiet=True):

    # Compress the image
    buffer = io.BytesIO()
    attacked_image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)

    # Update attacked_image with the compressed version
    attacked_image = Image.open(buffer)

    modes = ["psnr", "ssim", "nmi", "lpips"]

    results = {}
    for mode in modes:
        if mode in ["psnr", "ssim", "nmi"]:
            metrics = compute_image_distance_repeated(
                [clean_image],
                [attacked_image],
                metric_name=mode,
                num_workers=1,
                verbose=not quiet,
            )
            results[mode] = metrics

        elif mode == "lpips":
            model = load_perceptual_models(
                mode,
                mode="alex",
                device="cpu",
            )
            metrics = compute_perceptual_metric_repeated(
                [clean_image],
                [attacked_image],
                metric_name=mode,
                mode="alex",
                model=model,
                device="cpu",
            )
            results[mode] = metrics

    normalized_quality = 0
    for key, value in results.items():
        normalized_quality += (value[0] - QUALITY_OFFSETS[key]) * QUALITY_COEFFICIENTS[
            key
        ]
    return normalized_quality
