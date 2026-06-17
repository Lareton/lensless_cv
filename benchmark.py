import hydra

from src.runner import run_benchmark


@hydra.main(version_base=None, config_path="src/configs", config_name="benchmark")
def main(config):
    run_benchmark(config)


if __name__ == "__main__":
    main()
