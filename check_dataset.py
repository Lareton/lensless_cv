import hydra

from src.datasets.check import check_dataset


@hydra.main(version_base=None, config_path="src/configs", config_name="check_dataset")
def main(config):
    check_dataset(config)


if __name__ == "__main__":
    main()
