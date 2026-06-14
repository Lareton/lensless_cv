import os
from pathlib import Path


class CometLogger:
    def __init__(
        self,
        project_name,
        workspace=None,
        experiment_name=None,
        enabled=True,
        log_checkpoints=True,
    ):
        self.experiment = None
        self.log_checkpoints = log_checkpoints
        if not enabled:
            return

        api_key = os.getenv("COMET_API_KEY")
        if api_key is None:
            print("COMET_API_KEY is not set, Comet logging is disabled")
            return

        from comet_ml import Experiment

        self.experiment = Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
        )
        if experiment_name is not None:
            self.experiment.set_name(experiment_name)

    def log_parameters(self, parameters):
        if self.experiment is not None:
            self.experiment.log_parameters(parameters)

    def log_metrics(self, metrics, prefix=None, step=None):
        if self.experiment is None:
            return
        self.experiment.log_metrics(metrics, prefix=prefix, step=step)

    def log_images(self, paths, step=None):
        if self.experiment is None:
            return
        for path in paths:
            path = Path(path)
            self.experiment.log_image(
                image_data=str(path),
                name=path.stem,
                step=step,
            )

    def log_checkpoint(self, path):
        if self.experiment is not None and self.log_checkpoints:
            self.experiment.log_model(path.stem, str(path), overwrite=True)

    def end(self):
        if self.experiment is not None:
            self.experiment.end()
