import logging
import os

from omegaconf import errors
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch

from t3sc import models
from t3sc.data import DataModule
from t3sc.utils import Tester
from t3sc.callbacks import Backtracking

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device in use : {device}")

    # Fix seed for reproducibility
    logger.info(f"Using random seed {cfg.seed}")
    pl.seed_everything(cfg.seed)

    # Load datamodule
    datamodule = DataModule(**cfg.data.params)

    # Logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="tb", name="", version=""
    )

    # Callbacks
    callbacks = [
        cb.ModelCheckpoint(**cfg.checkpoint),
        cb.ModelCheckpoint(**cfg.checkpoint_best),
        cb.LearningRateMonitor(),
        cb.ProgressBar(),
    ]
    try:
        logger.info("Loading backtracking config")
        callbacks.append(Backtracking(**cfg.model.backtracking))
        logger.info("Backtracking callback instantiated successfully")
    except (errors.ConfigAttributeError, TypeError):
        logger.info("Backtracking config not found")

    # Instantiate model
    model_class = models.__dict__[cfg.model.class_name]
    model = model_class(**cfg.model.params).to(device)

    path_to_parameters = r"C:\Users\jerom\MCE\computational_imaging\projet\T3SC\data\trainings\2024-03-12_14-45-47\T3SC_dcmall_UniformNoise-mi0-ma55_beta1_ssl0_seed0\backtracking\state.pth"
    if os.path.exists(path_to_parameters):
        logger.info(f"Loading model parameters from: {path_to_parameters}")
        model.load_state_dict(torch.load(path_to_parameters)["state_dict"])
    else:
        logger.error(f"Model parameters not found at: {path_to_parameters}")

    path_to_metrics = r"C:\Users\jerom\MCE\computational_imaging\projet\T3SC\data\trainings\2024-03-12_14-45-47\T3SC_dcmall_UniformNoise-mi0-ma55_beta1_ssl0_seed0\backtracking\metrics.pth"
    if os.path.exists(path_to_metrics):
        logger.info(f"Loading metrics from: {path_to_metrics}")
        metrics = torch.load(path_to_metrics)
    else:
        logger.error(f"Metrics not found at: {path_to_metrics}")

    print("###########################################################################################################")
    print(type(metrics))
    print(metrics)
    print("###########################################################################################################")

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="tb", name="", version=""
    )

    # Instantiate trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=tb_logger,
        progress_bar_refresh_rate=0,
        **cfg.trainer.params,
    )

    # Print model info
    model.count_params()
    #
    # for data in enumerate(datamodule):
    #     print(data)

    # Fit trainer
    trainer.fit(model, datamodule=datamodule)

    # Load best checkpoint

    filename_best = os.listdir("best")[0]
    path_best = os.path.join("best", filename_best)
    logger.info(f"Loading best model for testing : {path_best}")
    model.load_state_dict(torch.load(path_best)["state_dict"])

    tester = Tester(**cfg.test)
    tester.eval(model, datamodule=datamodule)
