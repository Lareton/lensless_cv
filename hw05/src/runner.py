import random
import statistics
import time
from pathlib import Path

from omegaconf import OmegaConf


def set_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_train(config):
    set_seed(config.seed)
    logger = build_logger(config)
    logger.log_parameters(OmegaConf.to_container(config, resolve=True))
    print(OmegaConf.to_yaml(config, resolve=True))
    logger.end()


def run_inference(config):
    import numpy as np
    import torch
    from PIL import Image

    from src.datasets import build_dataloaders
    from src.datasets.preprocessing import crop_roi
    from src.metrics import ImageMetricTracker
    from src.utils.images import save_qualitative_grid

    set_seed(config.seed)
    logger = build_logger(config)
    logger.log_parameters(OmegaConf.to_container(config, resolve=True))
    device = get_device(config.device)
    model = build_model(config.model, config.checkpoint, device)
    dataloaders = build_dataloaders(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_tracker = None
    qualitative_left = config.qualitative_count
    model.eval()

    with torch.no_grad():
        for dataloader in dataloaders.values():
            for batch_idx, batch in enumerate(dataloader):
                model_batch = move_tensors(batch, device)
                output = model(**model_batch)
                reconstruction = output["reconstruction"]

                for image_id, sample in zip(
                    batch["image_id"],
                    reconstruction,
                ):
                    image = sample.detach().cpu().permute(1, 2, 0).numpy()
                    image = np.round(image.clip(0, 1) * 255).astype(np.uint8)
                    Image.fromarray(image).save(output_dir / f"{image_id}.png")

                if config.calculate_metrics and "target_roi" in model_batch:
                    if metric_tracker is None:
                        metric_tracker = ImageMetricTracker(config.metrics, device)
                    metric_tracker.update(
                        crop_roi(output["reconstruction"]),
                        model_batch["target_roi"],
                    )

                if qualitative_left > 0:
                    current_count = min(qualitative_left, reconstruction.shape[0])
                    paths = save_qualitative_grid(
                        batch,
                        reconstruction,
                        config.qualitative_dir,
                        current_count,
                    )
                    logger.log_images(paths)
                    qualitative_left -= current_count

                if (
                    config.max_batches is not None
                    and batch_idx + 1 >= config.max_batches
                ):
                    break

    if metric_tracker is not None:
        result = metric_tracker.compute()
        result["count"] = metric_tracker.count
        logger.log_metrics(result, prefix="test")
        print(OmegaConf.to_yaml(result))
    logger.end()


def run_metrics(config):
    from src.metrics import evaluate_directories

    logger = build_logger(config)
    logger.log_parameters(OmegaConf.to_container(config, resolve=True))
    device = get_device(config.device)
    result = evaluate_directories(
        config.ground_truth_dir,
        config.reconstruction_dir,
        config.metrics,
        device,
    )
    logger.log_metrics(result, prefix="test")
    print(OmegaConf.to_yaml(result))
    logger.end()


def run_benchmark(config):
    import torch

    from src.datasets import build_dataloaders

    set_seed(config.seed)
    logger = build_logger(config)
    logger.log_parameters(OmegaConf.to_container(config, resolve=True))
    device = get_device(config.device)
    model = build_model(config.model, config.checkpoint, device)
    dataloaders = build_dataloaders(config)
    dataloader = next(iter(dataloaders.values()))
    model.eval()

    with torch.no_grad():
        if config.warmup_batches > 0:
            for batch_idx, batch in enumerate(dataloader):
                model(**move_tensors(batch, device))
                synchronize(device)
                if batch_idx + 1 >= config.warmup_batches:
                    break

        times = []
        batch_sizes = []
        for batch_idx, batch in enumerate(dataloader):
            model_batch = move_tensors(batch, device)
            synchronize(device)
            start = time.perf_counter()
            model(**model_batch)
            synchronize(device)
            times.append(time.perf_counter() - start)
            batch_sizes.append(batch["lensless"].shape[0])

            if config.max_batches is not None and batch_idx + 1 >= config.max_batches:
                break

    if not times:
        raise ValueError("No batches were measured")

    total_images = sum(batch_sizes)
    total_time = sum(times)
    per_image = [
        elapsed / batch_size for elapsed, batch_size in zip(times, batch_sizes)
    ]
    result = {
        "batches": len(times),
        "images": total_images,
        "mean_ms": statistics.mean(per_image) * 1000,
        "std_ms": statistics.pstdev(per_image) * 1000,
        "median_ms": statistics.median(per_image) * 1000,
        "images_per_second": total_images / total_time,
        "device": str(device),
    }
    logger.log_metrics(result, prefix="benchmark")
    print(OmegaConf.to_yaml(result))
    logger.end()


def get_device(device):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_model(config, checkpoint, device):
    import torch
    from hydra.utils import instantiate

    model = instantiate(config).to(device)
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
        state_dict = checkpoint.get(
            "state_dict",
            checkpoint.get("model_state_dict", checkpoint),
        )
        model.load_state_dict(state_dict)
    return model


def move_tensors(batch, device):
    import torch

    return {
        key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)
    }


def synchronize(device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_logger(config):
    from hydra.utils import instantiate

    return instantiate(config.writer)
