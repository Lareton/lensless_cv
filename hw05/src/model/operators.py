import torch

from src.utils.padding import center_crop, center_pad


def prepare_psf(psf, shape):
    if psf.ndim == 3:
        psf = psf.unsqueeze(0)
    if psf.ndim != 4:
        raise ValueError(f"Expected BCHW or CHW PSF, got {tuple(psf.shape)}")

    psf = center_pad(psf, shape)
    psf = torch.fft.ifftshift(psf, dim=(-2, -1))
    return torch.fft.rfft2(psf)


def convolve_fft(x, psf_fft):
    return torch.fft.irfft2(torch.fft.rfft2(x) * psf_fft, s=x.shape[-2:])


def convolve_adjoint_fft(x, psf_fft):
    return torch.fft.irfft2(
        torch.fft.rfft2(x) * psf_fft.conj(),
        s=x.shape[-2:],
    )


def sensor_crop(x, shape):
    return center_crop(x, shape)


def sensor_adjoint(x, shape):
    return center_pad(x, shape)


def sensor_mask(sensor_shape, work_shape, reference):
    mask = reference.new_ones((*reference.shape[:-2], *sensor_shape))
    return sensor_adjoint(mask, work_shape)


def finite_difference(x):
    horizontal = torch.roll(x, shifts=-1, dims=-1) - x
    vertical = torch.roll(x, shifts=-1, dims=-2) - x
    return torch.stack((horizontal, vertical), dim=-3)


def finite_difference_adjoint(x):
    horizontal = torch.roll(x[..., 0, :, :], shifts=1, dims=-1) - x[..., 0, :, :]
    vertical = torch.roll(x[..., 1, :, :], shifts=1, dims=-2) - x[..., 1, :, :]
    return horizontal + vertical


def finite_difference_spectrum(shape, reference):
    height, width = shape
    fy = torch.fft.fftfreq(height, device=reference.device, dtype=reference.dtype)
    fx = torch.fft.rfftfreq(width, device=reference.device, dtype=reference.dtype)
    spectrum = 4 * torch.sin(torch.pi * fy).square().unsqueeze(-1)
    spectrum = spectrum + 4 * torch.sin(torch.pi * fx).square().unsqueeze(0)
    return spectrum.unsqueeze(0).unsqueeze(0)


def soft_threshold(x, threshold):
    return x.sign() * (x.abs() - threshold).clamp_min(0)


def project_nonnegative(x):
    return x.clamp_min(0)
