import hydra

from src.runner import run_metrics


@hydra.main(version_base=None, config_path="src/configs", config_name="calculate_metrics")
def main(config):
    run_metrics(config)


if __name__ == "__main__":
    main()
