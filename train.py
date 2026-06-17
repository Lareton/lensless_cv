import hydra

from src.runner import run_train


@hydra.main(version_base=None, config_path="src/configs", config_name="train")
def main(config):
    run_train(config)


if __name__ == "__main__":
    main()
