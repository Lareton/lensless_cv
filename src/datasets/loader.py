from hydra.utils import instantiate


def build_dataloaders(config):
    datasets = instantiate(config.datasets)
    loaders = {}

    for split, dataset in datasets.items():
        loaders[split] = instantiate(
            config.dataloader,
            dataset=dataset,
            shuffle=split == "train",
            drop_last=split == "train",
        )

    return loaders
