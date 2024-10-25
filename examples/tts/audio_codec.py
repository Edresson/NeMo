# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict

try:
    from one_logger_utils.pytorch_lightning import OneLoggerPTLTrainer
    from one_logger_utils.nemo import get_hooked_model
    import os
    HAVE_ONE_LOGGER = True
except ModuleNotFoundError:
    HAVE_ONE_LOGGER = False

from nemo.collections.tts.models import AudioCodecModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf/audio_codec", config_name="audio_codec")
def main(cfg):
    logging.info('\nConfig Params:\n%s', OmegaConf.to_yaml(cfg, resolve=True))
    if HAVE_ONE_LOGGER:
        # get exp name from wandb_logger_kwargs or from the log dir
        if "wandb_logger_kwargs" in cfg.exp_manager:
            exp_name = str(cfg.exp_manager.wandb_logger_kwargs.name)
        else:
            exp_name = cfg.log_dir.split(os.sep)[-2]

        # only include precision and num nodes
        one_logger_callback_config = {
            "enable_for_current_rank": os.environ.get('RANK') == '0',
            "one_logger_async": cfg.get("exp_manager").get("create_wandb_logger", False),
            "log_every_n_train_iterations": cfg.get("trainer").get("log_every_n_steps", 10),
            "app_tag_run_version": "0.0.0",
            "summary_data_schema_version": "1.0.0",
            "app_run_type": "training",
            "app_tag": exp_name,  # Please change this
            "app_tag_run_name": f"{exp_name}",  # Please change this
            "one_logger_project": "nemo-codec-train",  # Please change this
            "one_logger_run_name": exp_name,  # Please change this
            "world_size": os.environ.get('WORLD_SIZE', -1),
            "global_batch_size": cfg.get("model").get("train_ds").get("batch_size", 1),
            "batch_size": cfg.get("model").get("train_ds").get("batch_size", 1),
            "train_iterations_target": cfg.get("trainer").get("max_steps", 1),
            "train_samples_target": cfg.get("trainer").get("max_steps", 1)
            * cfg.get("model").get("train_ds").get("batch_size", 1),
            "is_train_iterations_enabled": True,
            "is_baseline_run": False,
            "is_test_iterations_enabled": False,
            "is_validation_iterations_enabled": True,
            "is_save_checkpoint_enabled": True,
            "is_log_throughput_enabled": False,
            "micro_batch_size": cfg.get("model").get("train_ds").get("batch_size", 1),
            "seq_length": 1,
            "save_checkpoint_strategy": "sync",
        }
        # if "callbacks" not in cfg.trainer:
        #     with open_dict(cfg):
        #         cfg.trainer.callbacks = []

        trainer = OneLoggerPTLTrainer(trainer_config=cfg.trainer, callback_config=one_logger_callback_config)
    else:
        trainer = pl.Trainer(**cfg.trainer)

    exp_manager(trainer, cfg.get("exp_manager", None))

    if HAVE_ONE_LOGGER:
        model = get_hooked_model(model_class=AudioCodecModel, cfg=cfg.model, trainer=trainer)
    else:
        model = AudioCodecModel(cfg=cfg.model, trainer=trainer)

    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
    trainer.fit(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
