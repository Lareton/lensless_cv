import torch
from omegaconf import OmegaConf

from src.utils.checkpoint import load_checkpoint, to_plain_container


def test_load_checkpoint_with_omegaconf_list(tmp_path):
    path = tmp_path / "checkpoint.pt"
    payload = {
        "state_dict": {"weight": torch.ones(1)},
        "optimizer_state_dict": {
            "param_groups": [{"betas": OmegaConf.create([0.9, 0.999])}]
        },
    }
    torch.save(payload, path)

    checkpoint = load_checkpoint(path)

    assert torch.equal(checkpoint["state_dict"]["weight"], torch.ones(1))
    assert checkpoint["optimizer_state_dict"]["param_groups"][0]["betas"][0] == 0.9


def test_plain_container_removes_omegaconf():
    value = {
        "optimizer_state_dict": {
            "param_groups": [{"betas": OmegaConf.create([0.9, 0.999])}]
        }
    }

    result = to_plain_container(value)

    assert result["optimizer_state_dict"]["param_groups"][0]["betas"] == [
        0.9,
        0.999,
    ]
