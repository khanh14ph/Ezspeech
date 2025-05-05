import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch


pl.seed_everything(42, workers=True)
torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_path="config/train", config_name="oov-asr")
def main(config: DictConfig):
    task = instantiate(config.task, _recursive_=False)
    if config.task.model.get("pretrained_weights") is not None:
        checkpoint_filepath = config.task.model.pretrained_weights
        checkpoint = torch.load(checkpoint_filepath, map_location="cpu")
        for attr, weights in checkpoint["state_dict"].items():
            if hasattr(task, attr):
                try:
                    net = getattr(task, attr)
                    net.load_state_dict(weights)
                except:
                    print(
                        f"***** Can't load {attr} from {checkpoint_filepath :<20s} *****"
                    )
                    continue
                print(f"***** Loading {attr} from {checkpoint_filepath :<20s} *****")

    callbacks = None
    if config.get("callbacks") is not None:
        callbacks = [instantiate(cfg) for _, cfg in config.callbacks.items()]

    loggers = None
    if config.get("loggers") is not None:
        loggers = [instantiate(cfg) for _, cfg in config.loggers.items()]

    trainer = pl.Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    trainer.fit(
        task,
        # ckpt_path="/data/khanhnd65/lightning_logs/oov/concat/checkpoints/last.ckpt",
    )


if __name__ == "__main__":
    main()