import torch

from src.model import DRUNet, LeADMM, ModularADMM


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def test_drunet_shape():
    model = DRUNet(
        channels=(4, 8, 12, 16),
        blocks=1,
        gradient_checkpointing=False,
    )
    image = torch.rand(2, 3, 17, 19)

    output = model(image)

    assert output.shape == image.shape


def test_processor_parameter_counts():
    pre8 = DRUNet(channels=(32, 64, 128, 256))
    pre4 = DRUNet(channels=(32, 64, 116, 128))

    assert parameter_count(pre8) == 8_167_299
    assert parameter_count(pre4) == 4_055_851
    assert 8_000_000 < 2 * parameter_count(pre4) < 8_200_000


def test_modular_admm_gradients():
    preprocessor = DRUNet(
        channels=(4, 8, 12, 16),
        blocks=1,
        gradient_checkpointing=True,
    )
    inversion = LeADMM(
        iterations=2,
        work_scale=1,
        gradient_checkpointing=True,
    )
    postprocessor = DRUNet(
        channels=(4, 8, 12, 16),
        blocks=1,
        gradient_checkpointing=True,
    )
    model = ModularADMM(inversion, preprocessor, postprocessor)
    measurement = torch.rand(1, 3, 17, 19)
    psf = torch.zeros_like(measurement)
    psf[..., 8, 9] = 1

    output = model(measurement, psf)["reconstruction"]
    output.square().mean().backward()

    assert output.shape == measurement.shape
    assert preprocessor.head.weight.grad is not None
    assert inversion.steps[-1].log_mu.grad is not None
    assert postprocessor.tail.weight.grad is not None
    assert len(model.parameters_by_iteration()) == 2
