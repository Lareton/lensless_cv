# Проект выполнен в рамках курса "Глубокое обучение AI360"

## Lensless Reconstruction

Выполнил: Ильтяков Никита

## Results

`inference ms/image` is the average wall-clock time per test image from the full
`inference.py` run: Comet run duration divided by 1500 test samples.
`benchmark ms/image` is measured by `benchmark.py` as pure model inference speed.

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MSE ↓ | Inference ms/image ↓ | Benchmark ms/image ↓ | Train run | Test run | Benchmark run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| Pre4 + LeADMM5 + Post4 | 16.034 | 0.343 | 0.539 | 0.0301 | 234.7 | 47.4 | [train](https://www.comet.com/iltyakov-nik/lensless-reconstruction/8b9f95bc8ed94899b31d8e16dd9c381a) | [test](https://www.comet.com/iltyakov-nik/lensless-reconstruction/2c86915cc42344c5813b73377b4f6899) | [benchmark](https://www.comet.com/iltyakov-nik/lensless-reconstruction/fde4b800e1ab4c288cd26e4ca5b06c2e) |
| Pre8 + LeADMM5 | 13.815 | 0.183 | 0.646 | 0.0477 | 236.7 | 33.7 | [train](https://www.comet.com/iltyakov-nik/lensless-reconstruction/100c5d0abdee4e2287f0c5a7882cac2a) | [test](https://www.comet.com/iltyakov-nik/lensless-reconstruction/bc18c205f2f546c4a94dca715f8fa27c) | [benchmark](https://www.comet.com/iltyakov-nik/lensless-reconstruction/307ce9b4d3f84c6aaab92b5637c04bd3) |
| LeADMM5 + Post8 | 15.883 | 0.340 | 0.555 | 0.0309 | 239.3 | 33.9 | [train](https://www.comet.com/iltyakov-nik/lensless-reconstruction/be085c775d50442394b0fbf0cbb6fa39) | [test](https://www.comet.com/iltyakov-nik/lensless-reconstruction/b76eb7290c1941168414f0b0166e2735) | [benchmark](https://www.comet.com/iltyakov-nik/lensless-reconstruction/735357ec412f452cb2f444b8805b615a) |
| LeADMM20 | 11.812 | 0.253 | 0.796 | 0.0764 | 292.0 | 66.6 | [train](https://www.comet.com/iltyakov-nik/lensless-reconstruction/0b9c3ce61973405cb67308b49459b10a) | [test](https://www.comet.com/iltyakov-nik/lensless-reconstruction/4740d7759cfa49969fe809e9b6bc0af0) | [benchmark](https://www.comet.com/iltyakov-nik/lensless-reconstruction/ed1f787284e042e18f577ace2abe9936) |

Этот репозиторий содержит проектную реализацию методов восстановления изображений
для lensless computational imaging: классический ADMM, обучаемый LeADMM и
модульные варианты с pre/post-processing сетями.

Проект использует датасет
`DigiCam-Mirflickr-MultiMask-10K`:

- `train` для обучения
- часть `train` для validation
- `test` для финальной оценки

Основные метрики качества:

- PSNR
- SSIM
- LPIPS
- MSE

Также логируются реконструкции, ground truth, lensless measurements, PSF,
training loss, validation loss и итоговые test/benchmark метрики в Comet ML.

## Links

- Demo: [demo.ipynb](demo.ipynb)
- Веса моделей: [Lareton/lensless_models](https://huggingface.co/Lareton/lensless_models/tree/main)
- Comet ML project: [lensless-reconstruction](https://www.comet.com/iltyakov-nik/lensless-reconstruction)
- Датасет: [DigiCam-Mirflickr-MultiMask-10K](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K)
- Основная статья: [Towards Robust and Generalizable Lensless Imaging with Modular Learned Reconstruction](https://arxiv.org/abs/2502.01102)
- LeADMM статья: [Learned reconstructions for practical mask-based lensless imaging](https://arxiv.org/abs/1908.11502)

## Installation

```bash
git clone https://github.com/Lareton/lensless_cv.git
cd lensless_cv
pip install -r requirements.txt
```

Для логирования в Comet ML нужно передать ключ через переменную среды:

```bash
export COMET_API_KEY=your_comet_api_key
```

## Download Data

Скачать весь датасет, который используется для обучения и тестирования:

```bash
python3 download_dataset.py \
  output_dir=data/digicam
```

Скачать только train split:

```bash
python3 download_dataset.py \
  output_dir=data/digicam \
  split=train
```

Скачать только test split:

```bash
python3 download_dataset.py \
  output_dir=data/digicam \
  split=test
```

Датасет также можно скачать лениво прямо во время запуска:

```bash
python3 train.py \
  datasets.train.download_if_missing=true \
  datasets.validation.download_if_missing=true
```

После скачивания ожидаемый путь выглядит так:

```text
data/digicam
```

## Download Weights

Скачать веса с Hugging Face:

```bash
hf download Lareton/lensless_models \
  --local-dir saved
```


Финальный чекпоинт лучшей модели ожидается по пути:

```text
saved/pre4-leadmm5-post4-scale4/best.pt
```

## Training

Финальная модель `Pre4 + LeADMM5 + Post4` была запущена такой командой:

```bash
COMET_API_KEY=$COMET_API_KEY python3 train.py \
  model=pre4_leadmm5_post4 \
  trainer.save_dir=saved/pre4-leadmm5-post4-scale4 \
  writer.experiment_name=pre4-leadmm5-post4-scale4
```

Остальные обучаемые варианты:

```bash
COMET_API_KEY=$COMET_API_KEY python3 train.py \
  model=pre8_leadmm5 \
  trainer.save_dir=saved/pre8-leadmm5-scale4 \
  writer.experiment_name=pre8-leadmm5-scale4
```

```bash
COMET_API_KEY=$COMET_API_KEY python3 train.py \
  model=leadmm5_post8 \
  trainer.save_dir=saved/leadmm5-post8-scale4 \
  writer.experiment_name=leadmm5-post8-scale4
```

```bash
COMET_API_KEY=$COMET_API_KEY python3 train.py \
  model=leadmm20 \
  trainer.save_dir=saved/leadmm20-scale4 \
  writer.experiment_name=leadmm20-scale4
```

## Inference And Test Metrics

Запуск инференса лучшей модели на test split:

```bash
COMET_API_KEY=$COMET_API_KEY python3 inference.py \
  model=pre4_leadmm5_post4 \
  checkpoint=saved/pre4-leadmm5-post4-scale4/best.pt \
  output_dir=data/final/pre4-leadmm5-post4 \
  writer.experiment_name=pre4-leadmm5-post4-test
```

Инференс сохраняет реконструкции в:

```text
data/final/pre4-leadmm5-post4
```

Итоговые метрики логируются в Comet ML с префиксом `test_`.

ADMM-100 без обучения можно запустить так:

```bash
COMET_API_KEY=$COMET_API_KEY python3 inference.py \
  model=admm100 \
  output_dir=data/final/admm100 \
  writer.experiment_name=admm100-test
```

Для внешней директории поддерживается формат:

```text
custom_data
├── lensless
├── masks
└── lensed
```

Запуск на такой директории:

```bash
COMET_API_KEY=$COMET_API_KEY python3 inference.py \
  datasets=custom_dir \
  datasets.test.data_dir=custom_data \
  model=pre4_leadmm5_post4 \
  checkpoint=saved/pre4-leadmm5-post4-scale4/best.pt \
  output_dir=data/custom_reconstructions \
  writer.experiment_name=custom-pre4-leadmm5-post4
```

Если метрики нужно посчитать отдельно для двух директорий:

```bash
COMET_API_KEY=$COMET_API_KEY python3 calculate_metrics.py \
  ground_truth_dir=custom_data/lensed \
  reconstruction_dir=data/custom_reconstructions \
  writer.experiment_name=custom-metrics
```

## Benchmark

Замер скорости лучшей модели:

```bash
COMET_API_KEY=$COMET_API_KEY python3 benchmark.py \
  model=pre4_leadmm5_post4 \
  checkpoint=saved/pre4-leadmm5-post4-scale4/best.pt \
  writer.experiment_name=pre4-leadmm5-post4-benchmark
```
