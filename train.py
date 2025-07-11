import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from ezspeech.models.ASR import RNNT_CTC_Training
from ezspeech.utils import color
from tqdm import tqdm

pl.seed_everything(42, workers=True)
torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_path="config", config_name="asr")
def main(config: DictConfig):
    task = RNNT_CTC_Training(config)
    if config.model.get("model_pretrained") is not None:
        checkpoint_filepath = config.model.model_pretrained.path + "/model_weights.ckpt"
        checkpoint = torch.load(
            checkpoint_filepath, map_location="cpu", weights_only=False
        )
        for attr in config.model.model_pretrained.include:
            if attr not in checkpoint["state_dict"].keys():
                print(f"Module {attr} not exist in checkpoint")
                continue
            if hasattr(task, attr):
                weights = checkpoint["state_dict"][attr]
                # try:
                net = getattr(task, attr).train()
                net.load_state_dict(weights)
                print(
                    f"Modules {color.GREEN}{attr}{color.RESET} loaded successfully from checkpoint"
                )
                # except:
                #     print(
                #         f"***** Can't load {color.RED}{attr}{color.RESET} from {checkpoint_filepath :<20s} *****"
                #     )
                #     continue

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
