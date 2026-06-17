import torch

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
    soft_threshold,
)
from src.utils.padding import center_crop, center_pad, fft_shape


def inner_product(x, y):
    return (x * y).sum()


def test_padding_and_crop():
    x = torch.randn(2, 3, 7, 8)
    padded = center_pad(x, (16, 18))

    assert padded.shape == (2, 3, 16, 18)
    assert torch.equal(center_crop(padded, x.shape[-2:]), x)
    assert fft_shape((380, 507)) == (768, 1024)


def test_sensor_adjoint():
    x = torch.randn(2, 3, 11, 13)
    y = torch.randn(2, 3, 6, 7)

    left = inner_product(sensor_crop(x, y.shape[-2:]), y)
    right = inner_product(x, sensor_adjoint(y, x.shape[-2:]))

    assert torch.allclose(left, right)


def test_convolution_adjoint():
    x = torch.randn(2, 3, 12, 14)
    y = torch.randn_like(x)
    psf = torch.randn(2, 3, 7, 9)
    psf_fft = prepare_psf(psf, x.shape[-2:])

    left = inner_product(convolve_fft(x, psf_fft), y)
    right = inner_product(x, convolve_adjoint_fft(y, psf_fft))

    assert torch.allclose(left, right, atol=1e-4, rtol=1e-4)


def test_identity_psf():
    x = torch.randn(2, 3, 12, 14)
    psf = torch.zeros(1, 3, 12, 14)
    psf[..., 6, 7] = 1
    psf_fft = prepare_psf(psf, x.shape[-2:])

    result = convolve_fft(x, psf_fft)

    assert torch.allclose(result, x, atol=1e-5, rtol=1e-5)


def test_psf_scale():
    psf = torch.rand(2, 3, 7, 9)
    psf_fft = prepare_psf(psf, (7, 9), scale=4)

    assert torch.allclose(
        psf_fft[..., 0, 0].real,
        torch.full((2, 3), 4.0),
        atol=1e-5,
    )


def test_finite_difference_adjoint():
    x = torch.randn(2, 3, 11, 13)
    y = torch.randn(2, 3, 2, 11, 13)

    left = inner_product(finite_difference(x), y)
    right = inner_product(x, finite_difference_adjoint(y))

    assert torch.allclose(left, right, atol=1e-5, rtol=1e-5)


def test_finite_difference_spectrum():
    x = torch.randn(2, 3, 12, 14)
    expected = finite_difference_adjoint(finite_difference(x))
    spectrum = finite_difference_spectrum(x.shape[-2:], x)
    result = torch.fft.irfft2(
        torch.fft.rfft2(x) * spectrum,
        s=x.shape[-2:],
    )

    assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5)


def test_proximal_operators():
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0])

    assert torch.equal(
        soft_threshold(x, 1),
        torch.tensor([-1.0, 0.0, 0.0, 1.0]),
    )
    assert torch.equal(
        project_nonnegative(x),
        torch.tensor([0.0, 0.0, 0.5, 2.0]),
    )
