# Проект выполнен в рамках курса "Глубокое обучение AI360"

## Lensless Reconstruction

Выполнил: Ильтяков Никита

Model weights: [Hugging Face](https://huggingface.co/Lareton/lensless_models/tree/main)

`inference ms/image` is the average wall-clock time per test image from the full
`inference.py` run: Comet run duration divided by 1500 test samples.
`benchmark ms/image` is measured by `benchmark.py` as pure model inference speed.

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MSE ↓ | Inference ms/image ↓ | Benchmark ms/image ↓ | Train run | Test run | Benchmark run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| Pre4 + LeADMM5 + Post4 | 16.034 | 0.343 | 0.539 | 0.0301 | 234.7 | 47.4 | [train](https://www.comet.com/iltyakov-nik/lensless-reconstruction/0b9c3ce61973405cb67308b49459b10a) | [test](https://www.comet.com/iltyakov-nik/lensless-reconstruction/2c86915cc42344c5813b73377b4f6899) | [benchmark](https://www.comet.com/iltyakov-nik/lensless-reconstruction/fde4b800e1ab4c288cd26e4ca5b06c2e) |
| Pre8 + LeADMM5 | 13.815 | 0.183 | 0.646 | 0.0477 | 236.7 | 33.7 | [train](https://www.comet.com/iltyakov-nik/lensless-reconstruction/be085c775d50442394b0fbf0cbb6fa39) | [test](https://www.comet.com/iltyakov-nik/lensless-reconstruction/bc18c205f2f546c4a94dca715f8fa27c) | [benchmark](https://www.comet.com/iltyakov-nik/lensless-reconstruction/307ce9b4d3f84c6aaab92b5637c04bd3) |
| LeADMM5 + Post8 | 15.883 | 0.340 | 0.555 | 0.0309 | 239.3 | 33.9 | [train](https://www.comet.com/iltyakov-nik/lensless-reconstruction/8b9f95bc8ed94899b31d8e16dd9c381a) | [test](https://www.comet.com/iltyakov-nik/lensless-reconstruction/b76eb7290c1941168414f0b0166e2735) | [benchmark](https://www.comet.com/iltyakov-nik/lensless-reconstruction/735357ec412f452cb2f444b8805b615a) |
| LeADMM20 | 11.812 | 0.253 | 0.796 | 0.0764 | 292.0 | 66.6 | [train](https://www.comet.com/iltyakov-nik/lensless-reconstruction/0b9c3ce61973405cb67308b49459b10a) | [test](https://www.comet.com/iltyakov-nik/lensless-reconstruction/4740d7759cfa49969fe809e9b6bc0af0) | [benchmark](https://www.comet.com/iltyakov-nik/lensless-reconstruction/ed1f787284e042e18f577ace2abe9936) |
