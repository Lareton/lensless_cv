import hydra

from src.runner import run_inference


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    run_inference(config)


if __name__ == "__main__":
    main()
