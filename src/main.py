import hydra
from omegaconf import DictConfig, OmegaConf
from build import Builder
from hydra.utils import instantiate

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    builder = Builder(cfg)
    builder.train_and_compare()

if __name__ == "__main__":
    main()