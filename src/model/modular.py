from torch import nn


class ModularADMM(nn.Module):
    def __init__(self, inversion, preprocessor=None, postprocessor=None):
        super().__init__()
        self.preprocessor = preprocessor
        self.inversion = inversion
        self.postprocessor = postprocessor

    def forward(self, lensless, psf, **batch):
        measurement = lensless
        if self.preprocessor is not None:
            measurement = self.preprocessor(measurement)

        output = self.inversion(measurement, psf, **batch)
        reconstruction = output["reconstruction"]
        if self.postprocessor is not None:
            reconstruction = self.postprocessor(reconstruction)

        output["reconstruction"] = reconstruction
        return output

    def parameters_by_iteration(self):
        return self.inversion.parameters_by_iteration()
