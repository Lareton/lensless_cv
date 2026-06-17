import torch


def collate_fn(items):
    batch = {}
    for key in items[0]:
        values = [item[key] for item in items]
        batch[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
    return batch
