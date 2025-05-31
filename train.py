import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from ezspeech.models.recognition import SpeechRecognitionTask
# from ezspeech.tasks.ctc_recognition import SpeechRecognitionTask
from tqdm import tqdm
pl.seed_everything(42, workers=True)
torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_path="config", config_name="asr1")
def main(config: DictConfig):
    task = SpeechRecognitionTask(config)
    # if config.model.get("pretrained_weights") is not None:
    #     checkpoint_filepath = config.model.pretrained_weights
    #     checkpoint = torch.load(checkpoint_filepath, map_location="cpu",weights_only=False)
    #     print(checkpoint["hyper_parameters"].items())
    #     for attr, _ in checkpoint["hyper_parameters"].items():
    #         weights=checkpoint["state_dict"][attr]
    #         print("attr:",attr)
    #         if hasattr(task, attr):
    #             net = getattr(task, attr)
    #             net.load_state_dict(weights)

    #             print(f"***** Loading {attr} from {checkpoint_filepath :<20s} *****")
    task.restore_from(config.model.get("pretrained_weights"))
    
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