from collections import namedtuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.model.operators import (
    convolve_adjoint_fft,
    convolve_fft,
    finite_difference,
    finite_difference_adjoint,
    finite_difference_spectrum,
    prepare_psf,
    project_nonnegative,
    sensor_adjoint,
    sensor_crop,
    sensor_mask,
    soft_threshold,
)
from src.utils.padding import fft_shape

ADMMState = namedtuple(
    "ADMMState",
    (
        "x",
        "u",
        "v",
        "w",
        "dual_v",
        "dual_u",
        "dual_w",
    ),
)


def init_admm_state(measurement, work_shape):
    batch, channels = measurement.shape[:2]
    shape = (batch, channels, *work_shape)
    x = measurement.new_zeros(shape)

    return ADMMState(
        x=x,
        u=measurement.new_zeros((batch, channels, 2, *work_shape)),
        v=x.clone(),
        w=x.clone(),
        dual_v=x.clone(),
        dual_u=measurement.new_zeros((batch, channels, 2, *work_shape)),
        dual_w=x.clone(),
    )


class FixedADMMIteration(nn.Module):
    def __init__(self, mu=(1e-4, 1e-4, 1e-4), tau=2e-4):
        super().__init__()
        self.register_buffer("_mu", torch.tensor(mu))
        self.register_buffer("_tau", torch.tensor(tau))

    @property
    def mu(self):
        return self._mu

    @property
    def tau(self):
        return self._tau

    def threshold(self, mu2):
        return self.tau / mu2

    def forward(
        self,
        measurement,
        psf_fft,
        state,
        padded_measurement=None,
        mask=None,
        denominator=None,
    ):
        mu1, mu2, mu3 = self.mu.to(state.x)
        sensor_shape = measurement.shape[-2:]
        work_shape = state.x.shape[-2:]

        if padded_measurement is None:
            padded_measurement = sensor_adjoint(measurement, work_shape)
        if mask is None:
            mask = sensor_mask(sensor_shape, work_shape, state.x)
        if denominator is None:
            denominator = mu1 * psf_fft.abs().square() + mu3
            denominator = denominator + mu2 * finite_difference_spectrum(
                work_shape,
                state.x,
            )

        gradient = finite_difference(state.x)
        u = soft_threshold(
            gradient + state.dual_u / mu2,
            self.threshold(mu2).to(state.x),
        )

        hx = convolve_fft(state.x, psf_fft)
        v = (state.dual_v + mu1 * hx + padded_measurement) / (mu1 + mask)

        w = project_nonnegative(state.x + state.dual_w / mu3)

        rhs = mu3 * w - state.dual_w
        rhs = rhs + finite_difference_adjoint(mu2 * u - state.dual_u)
        rhs = rhs + convolve_adjoint_fft(mu1 * v - state.dual_v, psf_fft)

        x = torch.fft.irfft2(
            torch.fft.rfft2(rhs) / denominator,
            s=work_shape,
        )

        dual_v = state.dual_v + mu1 * (convolve_fft(x, psf_fft) - v)
        dual_u = state.dual_u + mu2 * (finite_difference(x) - u)
        dual_w = state.dual_w + mu3 * (x - w)

        return ADMMState(
            x=x,
            u=u,
            v=v,
            w=w,
            dual_v=dual_v,
            dual_u=dual_u,
            dual_w=dual_w,
        )


class TrainableADMMIteration(FixedADMMIteration):
    def __init__(self, mu=(1e-4, 1e-4, 1e-4), tau=2e-4):
        super().__init__(mu, tau)
        del self._mu
        del self._tau
        self.log_mu = nn.Parameter(torch.tensor(mu).log())
        self.log_tau = nn.Parameter(torch.tensor(float(tau)).log())

    @property
    def mu(self):
        return self.log_mu.exp()

    @property
    def tau(self):
        return self.log_tau.exp()

    def threshold(self, mu2):
        return self.tau


class ADMM(nn.Module):
    def __init__(
        self,
        iterations=100,
        mu=(1e-4, 1e-4, 1e-4),
        tau=2e-4,
        work_scale=2,
        psf_scale=1.0,
        return_history=False,
        trainable=False,
    ):
        super().__init__()
        if trainable:
            raise ValueError("Use LeADMM for trainable hyperparameters")
        if iterations < 1:
            raise ValueError("iterations must be positive")

        self.iterations = iterations
        self.work_scale = work_scale
        self.psf_scale = psf_scale
        self.return_history = return_history
        self.step = FixedADMMIteration(mu, tau)

    def forward(self, lensless, psf, **batch):
        sensor_shape = lensless.shape[-2:]
        work_shape = fft_shape(sensor_shape, self.work_scale)
        psf_fft = prepare_psf(psf, work_shape, self.psf_scale)
        state = init_admm_state(lensless, work_shape)

        mu1, mu2, mu3 = self.step.mu.to(lensless)
        padded_measurement = sensor_adjoint(lensless, work_shape)
        mask = sensor_mask(sensor_shape, work_shape, state.x)
        denominator = mu1 * psf_fft.abs().square() + mu3
        denominator = denominator + mu2 * finite_difference_spectrum(
            work_shape,
            state.x,
        )

        history = []
        if self.return_history:
            history.append(data_fidelity(lensless, state.x, psf_fft))

        for _ in range(self.iterations):
            state = self.step(
                lensless,
                psf_fft,
                state,
                padded_measurement,
                mask,
                denominator,
            )
            if self.return_history:
                history.append(data_fidelity(lensless, state.x, psf_fft))

        reconstruction = sensor_crop(state.x, sensor_shape).clamp(0, 1)
        output = {"reconstruction": reconstruction}

        if self.return_history:
            output["data_fidelity"] = torch.stack(history)

        return output


class LeADMM(nn.Module):
    def __init__(
        self,
        iterations=20,
        mu=(1e-4, 1e-4, 1e-4),
        tau=2e-4,
        work_scale=2,
        psf_scale=1.0,
        gradient_checkpointing=True,
    ):
        super().__init__()
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if any(value <= 0 for value in mu) or tau <= 0:
            raise ValueError("ADMM parameters must be positive")

        self.iterations = iterations
        self.work_scale = work_scale
        self.psf_scale = psf_scale
        self.gradient_checkpointing = gradient_checkpointing
        self.steps = nn.ModuleList(
            TrainableADMMIteration(mu, tau) for _ in range(iterations)
        )

    def forward(self, lensless, psf, **batch):
        sensor_shape = lensless.shape[-2:]
        work_shape = fft_shape(sensor_shape, self.work_scale)
        psf_fft = prepare_psf(psf, work_shape, self.psf_scale)
        state = init_admm_state(lensless, work_shape)
        padded_measurement = sensor_adjoint(lensless, work_shape)
        mask = sensor_mask(sensor_shape, work_shape, state.x)

        for step in self.steps:
            if self.training and self.gradient_checkpointing:
                state = self._checkpoint_step(
                    step,
                    lensless,
                    psf_fft,
                    state,
                    padded_measurement,
                    mask,
                )
            else:
                state = step(
                    lensless,
                    psf_fft,
                    state,
                    padded_measurement,
                    mask,
                )

        reconstruction = sensor_crop(state.x, sensor_shape)
        return {"reconstruction": reconstruction}

    @staticmethod
    def _checkpoint_step(
        step,
        measurement,
        psf_fft,
        state,
        padded_measurement,
        mask,
    ):
        def run(*values):
            current = ADMMState(*values)
            return step(
                measurement,
                psf_fft,
                current,
                padded_measurement,
                mask,
            )

        return checkpoint(run, *state, use_reentrant=False)

    def parameters_by_iteration(self):
        return [
            {
                "mu1": step.mu[0].item(),
                "mu2": step.mu[1].item(),
                "mu3": step.mu[2].item(),
                "tau": step.tau.item(),
            }
            for step in self.steps
        ]


def data_fidelity(measurement, reconstruction, psf_fft):
    prediction = sensor_crop(
        convolve_fft(reconstruction, psf_fft),
        measurement.shape[-2:],
    )
    error = prediction - measurement
    return 0.5 * error.square().flatten(1).sum(dim=1)
