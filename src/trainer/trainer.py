from pathlib import Path

import torch
from hydra.utils import instantiate

from src.datasets.preprocessing import crop_roi
from src.metrics import ImageMetricTracker
from src.utils.checkpoint import (
    load_checkpoint as load_torch_checkpoint,
    to_plain_container,
)
from src.utils.images import save_qualitative_grid


class Trainer:
    def __init__(
        self,
        model,
        loss,
        optimizer,
        dataloaders,
        metrics,
        logger,
        device,
        config,
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.metrics = metrics
        self.logger = logger
        self.device = device
        self.config = config
        self.amp = config.amp and device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.epoch = 0
        self.global_step = 0
        self.best_psnr = float("-inf")

        if config.resume_from is not None:
            self.load_checkpoint(config.resume_from)

    def fit(self):
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            self.train_epoch()
            result = self.validate()
            self.log_parameters()

            if result["psnr"] > self.best_psnr:
                self.best_psnr = result["psnr"]
                self.save_checkpoint("best.pt")
            self.save_checkpoint("last.pt")

    def train_epoch(self):
        self.model.train()
        self.loss.train()
        if hasattr(self.loss.lpips, "reset"):
            self.loss.lpips.reset()
        self.optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "mse": 0.0, "lpips": 0.0}
        train_loader = self.dataloaders["train"]
        accumulation = self.config.accumulation_steps

        for batch_idx, batch in enumerate(train_loader):
            progress = (
                self.epoch + batch_idx / max(len(train_loader), 1)
            ) / self.config.epochs
            self.loss.set_progress(progress)
            model_batch = move_tensors(batch, self.device)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.amp,
            ):
                output = self.model(**model_batch)
                values = self.loss(
                    output["reconstruction"],
                    model_batch["target_roi"],
                )
                scaled_loss = values["loss"] / accumulation

            self.scaler.scale(scaled_loss).backward()
            reached_limit = (
                self.config.max_train_batches is not None
                and batch_idx + 1 >= self.config.max_train_batches
            )
            should_step = (
                (batch_idx + 1) % accumulation == 0
                or batch_idx + 1 == len(train_loader)
                or reached_limit
            )
            if should_step:
                self.step_optimizer()

            for name in totals:
                totals[name] += values[name].detach().item()

            if (batch_idx + 1) % self.config.log_step == 0:
                result = {
                    name: value / self.config.log_step for name, value in totals.items()
                }
                result["learning_rate"] = self.optimizer.param_groups[0]["lr"]
                result["mse_weight"] = values["mse_weight"].item()
                result["lpips_weight"] = values["lpips_weight"].item()
                self.logger.log_metrics(
                    result,
                    prefix="train",
                    step=self.global_step,
                )
                print(
                    f"epoch={self.epoch + 1} "
                    f"batch={batch_idx + 1}/{len(train_loader)} "
                    f"loss={result['loss']:.6f}"
                )
                totals = {name: 0.0 for name in totals}

            if reached_limit:
                break

    def step_optimizer(self):
        if self.config.grad_clip is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.loss.eval()
        tracker = ImageMetricTracker(
            self.metrics,
            self.device,
            lpips=self.loss.lpips,
        )
        total_loss = 0.0
        batches = 0
        last_batch = None
        last_reconstruction = None

        for batch_idx, batch in enumerate(self.dataloaders["validation"]):
            model_batch = move_tensors(batch, self.device)
            output = self.model(**model_batch)
            values = self.loss(
                output["reconstruction"],
                model_batch["target_roi"],
            )
            tracker.update(
                crop_roi(output["reconstruction"]),
                model_batch["target_roi"],
            )
            total_loss += values["loss"].item()
            batches += 1
            last_batch = batch
            last_reconstruction = output["reconstruction"]

            if (
                self.config.validation_batches is not None
                and batch_idx + 1 >= self.config.validation_batches
            ):
                break

        if batches == 0:
            raise ValueError("Validation loader is empty")

        result = tracker.compute()
        result["loss"] = total_loss / batches
        self.logger.log_metrics(
            result,
            prefix="validation",
            step=self.global_step,
        )
        self.log_images(last_batch, last_reconstruction)
        print(
            f"epoch={self.epoch + 1} "
            f"validation_loss={result['loss']:.6f} "
            f"psnr={result['psnr']:.4f}"
        )
        return result

    def log_images(self, batch, reconstruction):
        output_dir = self.save_dir / "qualitative" / f"epoch_{self.epoch + 1:03d}"
        paths = save_qualitative_grid(
            batch,
            reconstruction,
            output_dir,
            self.config.qualitative_count,
        )
        self.logger.log_images(paths, step=self.global_step)

    def log_parameters(self):
        if not hasattr(self.model, "parameters_by_iteration"):
            return
        parameters = {}
        for index, values in enumerate(self.model.parameters_by_iteration()):
            for name, value in values.items():
                parameters[f"iteration_{index + 1:02d}/{name}"] = value
        self.logger.log_metrics(
            parameters,
            prefix="model",
            step=self.global_step,
        )

    def save_checkpoint(self, name):
        path = self.save_dir / name
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "epoch": self.epoch + 1,
            "global_step": self.global_step,
            "best_psnr": self.best_psnr,
        }
        torch.save(to_plain_container(checkpoint), path)
        self.logger.log_checkpoint(path)

    def load_checkpoint(self, path):
        checkpoint = load_torch_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_psnr = checkpoint["best_psnr"]


def build_trainer(config, model, dataloaders, logger, device):
    loss = instantiate(config.loss).to(device)
    optimizer = instantiate(config.optimizer, params=model.parameters())
    return Trainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        dataloaders=dataloaders,
        metrics=config.metrics,
        logger=logger,
        device=device,
        config=config.trainer,
    )


def move_tensors(batch, device):
    return {
        key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)
    }
