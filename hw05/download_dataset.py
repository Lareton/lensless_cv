import hydra

from src.datasets.download import download_digicam


@hydra.main(version_base=None, config_path="src/configs", config_name="download")
def main(config):
    download_digicam(config.output_dir, config.repo_id, config.split)


if __name__ == "__main__":
    main()
