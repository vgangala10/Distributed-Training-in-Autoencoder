import torch
from torch import optim
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from autoencoder import *
from Dataloader import *

path_model = './'

os.makedirs(path_model, exist_ok = True)

if __name__ == '__main__':
    logger = TensorBoardLogger("tb_logs", name="mnist_model_v1")
    torch.set_float32_matmul_precision('medium')
    strategy = DeepSpeedStrategy()
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(path_model+"/tb_logs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    )

    model = AELightningModule()

    world_size = torch.cuda.device_count()

    dm = AEdata(
        batch_size = 32,
        num_workers = 8
    )

    checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="Validation_epoch_loss",
    mode="min",
    dirpath=path_model,
    filename='best_model-{epoch:02d}'
)
    trainer = pl.Trainer(accelerator = "gpu",
                        devices = list(range(world_size)), 
                        max_epochs=100,
                        profiler = profiler,
                        strategy=DeepSpeedStrategy(logging_batch_size_per_gpu=32),
                        log_every_n_steps = 100,
                        default_root_dir=path_model,
                        callbacks = [checkpoint_callback])
    trainer.fit(model, dm)