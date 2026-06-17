import numpy as np


class SLMParam:
    NAME = "name"
    CELL_SIZE = "cell_dim"
    SHAPE = "shape"
    SIZE = "size"
    DEADSPACE = "deadspace"
    PITCH = "pitch"
    COLOR_FILTER = "color_filter"


slm_dict = {
    "adafruit": {
        SLMParam.NAME: "adafruit",
        SLMParam.CELL_SIZE: np.array([0.06e-3, 0.18e-3]),
        SLMParam.SHAPE: np.array([128 * 3, 160]),
        SLMParam.SIZE: np.array([28.03e-3, 35.04e-3]),
        SLMParam.COLOR_FILTER: np.array(
            [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        )[:, np.newaxis],
    }
}

for config in slm_dict.values():
    config[SLMParam.DEADSPACE] = (
        config[SLMParam.SIZE] - config[SLMParam.CELL_SIZE] * config[SLMParam.SHAPE]
    ) / (config[SLMParam.SHAPE] - 1)
    config[SLMParam.PITCH] = (
        config[SLMParam.CELL_SIZE] + config[SLMParam.DEADSPACE]
    )
