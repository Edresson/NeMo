# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import torch


class PROFILING(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.memory = False  # False -> nsys profile, True -> memory profile

        # Uncomment below for a very detailed profile with a lot of overhead.
        # It’s useful to create profile with and without NVTX and compare.
        #
        import nvtx

        self.pr = nvtx.Profile()
        # self.ctx = torch.autograd.profiler.emit_nvtx()

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch, batch_idx: int
    ) -> None:
        """Called when the train batch begins."""
        if batch_idx == 1:
            if self.memory:
                torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)
            else:
                # Uncomment if using NVTX.
                self.pr.enable()
                torch.autograd.profiler.emit_nvtx().__enter__()
                torch.cuda.profiler.cudart().cudaProfilerStart()

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int
    ) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.

        """
        if batch_idx == 4:
            if self.memory:
                torch.cuda.memory._dump_snapshot(f"/results/memory-snapshot.pickle")
                torch.cuda.memory._record_memory_history(enabled=None)
                import sys

                sys.exit(0)
            else:
                torch.cuda.profiler.cudart().cudaProfilerStop()
                # Uncomment if using NVTX.
                torch.autograd.profiler.emit_nvtx().__exit__(None, None, None)
                self.pr.disable()
                import sys

                sys.exit(0)


import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.multimodal.speech_llm.models.modular_models import ModularAudioGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager

mp.set_start_method("spawn", force=True)

"""
MEGATRON_CKPT=/path/to/megatron-llm.nemo
ASR_MODEL=/path/to/asr-model.nemo

TRAIN_MANIFESTS="[/data/train_1.json,/data/train_2.json]"
VAL_MANIFESTS="[/data/dev_1.json,/data/dev_2.json]"
VAL_NAMES="[dev-1,dev-2]"

CUDA_VISIBLE_DEVICES="0,1" python modular_audio_gpt_train.py --config-path="./conf" --config-name "modular_audio_gpt_config_peft" \
    trainer.devices=-1 \
    model.freeze_audio_encoder=True \
    model.freeze_llm=True \
    model.global_batch_size=4 \
    model.micro_batch_size=2 \
    model.pretrained_audio_model=$ASR_MODEL \
    model.restore_from_path=$MEGATRON_MODEL \
    model.data.train_ds.manifest_filepath=$TRAIN_MANIFESTS \
    model.data.validation_ds.manifest_filepath=$VAL_MANIFESTS \
    ++model.data.validation_ds.names=$VAL_NAMES \
"""


@hydra_runner(config_path="conf", config_name="modular_audio_gpt_config_peft")
def main(cfg) -> None:
    # Set up logging with the specified log level
    logging_level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    logging.setLevel(logging_level)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')
    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    precision = cfg.trainer.precision
    trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer()
    if hasattr(cfg, 'do_profiling') and cfg.do_profiling:
        trainer.callbacks.append(PROFILING())
    cfg.trainer.precision = precision

    exp_manager(trainer, cfg.exp_manager)
    # update resume from checkpoint found by exp_manager
    logging.info(f'Resuming training from checkpoint: {trainer.ckpt_path}')

    if hasattr(cfg, 'model_target'):
        imported_cls = model_utils.import_class_by_path(cfg.model_target)
    else:
        imported_cls = ModularAudioGPTModel
    model = imported_cls.restore_from_pretrained_models(cfg, trainer=trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()
