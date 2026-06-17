import torch

from src.model import (
    ADMM,
    FixedADMMIteration,
    LeADMM,
    init_admm_state,
    prepare_psf,
)
from src.model.operators import convolve_fft, sensor_crop
from src.utils.padding import center_pad, fft_shape


def test_admm_iteration():
    measurement = torch.rand(2, 3, 12, 14)
    work_shape = (24, 30)
    psf = torch.zeros(2, 3, *measurement.shape[-2:])
    psf[..., 6, 7] = 1
    psf_fft = prepare_psf(psf, work_shape)
    state = init_admm_state(measurement, work_shape)

    result = FixedADMMIteration()(measurement, psf_fft, state)

    assert result.x.shape == (2, 3, *work_shape)
    assert result.u.shape == (2, 3, 2, *work_shape)
    assert result.v.shape == result.x.shape
    assert result.w.shape == result.x.shape

    for value in result:
        assert torch.isfinite(value).all()
        assert not value.is_complex()


def test_admm_identity_psf():
    measurement = torch.rand(1, 3, 12, 14)
    psf = torch.zeros(1, 3, 12, 14)
    psf[..., 6, 7] = 1

    output = ADMM(iterations=100, tau=0, return_history=True)(
        lensless=measurement,
        psf=psf,
        target=torch.zeros_like(measurement),
    )

    assert output["reconstruction"].shape == measurement.shape
    assert output["data_fidelity"].shape == (101, 1)
    assert output["data_fidelity"][-1, 0] < output["data_fidelity"][0, 0]
    assert torch.mean((output["reconstruction"] - measurement).square()) < 1e-6


def test_admm_blur_fidelity():
    target = torch.zeros(1, 3, 16, 18)
    target[..., 4:12, 5:13] = 1
    psf = torch.zeros(1, 3, 9, 9)
    psf[..., 4, 3:6] = 1 / 3
    work_shape = fft_shape(target.shape[-2:])
    psf_fft = prepare_psf(psf, work_shape)
    measurement = sensor_crop(
        convolve_fft(center_pad(target, work_shape), psf_fft),
        target.shape[-2:],
    )

    output = ADMM(iterations=50, return_history=True)(measurement, psf)
    history = output["data_fidelity"][:, 0]

    assert history[-1] < history[0]
    assert torch.isfinite(output["reconstruction"]).all()


def test_leadmm_gradients():
    measurement = torch.rand(1, 3, 12, 14)
    psf = torch.zeros_like(measurement)
    psf[..., 6, 7] = 1
    model = LeADMM(iterations=3, gradient_checkpointing=True)

    output = model(measurement, psf)["reconstruction"]
    output.square().mean().backward()

    assert len(model.steps) == 3
    assert all((step.mu > 0).all() for step in model.steps)
    assert all(step.tau > 0 for step in model.steps)
    assert all(step.log_mu.grad is not None for step in model.steps)
    assert model.steps[-1].log_tau.grad is not None
    assert model.steps[-1].log_tau.grad.abs() > 0


def test_leadmm_checkpoint(tmp_path):
    measurement = torch.rand(1, 3, 12, 14)
    psf = torch.zeros_like(measurement)
    psf[..., 6, 7] = 1
    model = LeADMM(iterations=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    output = model(measurement, psf)["reconstruction"]
    output.square().mean().backward()
    optimizer.step()
    path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict()}, path)

    restored = LeADMM(iterations=2)
    checkpoint = torch.load(path, weights_only=True)
    restored.load_state_dict(checkpoint["state_dict"])

    model.eval()
    restored.eval()
    expected = model(measurement, psf)["reconstruction"]
    actual = restored(measurement, psf)["reconstruction"]
    assert torch.allclose(actual, expected)
