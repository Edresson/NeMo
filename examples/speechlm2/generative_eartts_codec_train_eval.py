# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Evaluation script for GenerativeCodecEARTTS reconstruction capabilities.

Args:
    config-path (str):
        Path to the directory containing the YAML configuration file.

    config-name (str):
        Name of the YAML configuration file.

    checkpoint_path (str):
        Path to the GenerativeCodecEARTTS checkpoint file (.ckpt).

    audio_dir (str):
        Path to the directory containing target `.wav` or `.flac` files to reconstruct.

    out_dir (str):
        Directory where reconstructed audio samples will be saved.

    prompt_duration (float):
        Seconds of audio to use from the beginning of each file as the 
        conditioning prompt (default: 3.0s).

    max_batches (int):
        Maximum number of batches to evaluate. Useful for debugging. 
        Defaults to -1 (evaluate all batches).

Usage:
    python eval_generative_codec.py \
        --config-path=conf/ \
        --config-name=duplex_eartts.yaml \
        ++checkpoint_path=results/model.ckpt \
        ++audio_dir=/path/to/test_audio \
        ++out_dir=results/reconstructed_samples \
        ++prompt_duration=3.0 \
        ++max_batches=2
"""

import os
import glob
from functools import partial

import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.asr.models import EncDecRNNTBPEModel

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from contextlib import nullcontext

from omegaconf import OmegaConf
from nemo.core.config import hydra_runner

# Import metrics
from nemo.collections.speechlm2.parts.metrics.asr_cer_wer import Intelligibility
from nemo.collections.speechlm2.parts.metrics.secs import SECS

# Custom module
from nemo.collections.speechlm2.models.generative_eartts_codec import GenerativeCodecEARTTS

if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


class AudioDirectoryDataset(Dataset):
    """
    Standard PyTorch Dataset for reading pure .wav and .flac files from a directory,
    recursively searching through all subfolders.
    """
    def __init__(self, audio_dir):
        # Search for both .wav and .flac files recursively using **
        wav_files = glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True)
        flac_files = glob.glob(os.path.join(audio_dir, "**", "*.flac"), recursive=True)
        
        self.files = wav_files + flac_files
        
        if len(self.files) == 0:
            raise ValueError(f"No .wav or .flac files found recursively in directory: {audio_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return {"audio_filepath": self.files[idx]}


def collate_audio_only(batch, sample_rate):
    """
    Loads audio files and pads them to the maximum length in the batch.
    """
    audio_list = []
    audio_lengths = []
    filepaths = []

    for s in batch:
        wav, _ = librosa.load(s["audio_filepath"], sr=sample_rate, mono=True)
        wav = torch.as_tensor(wav, dtype=torch.float32)
        audio_list.append(wav)
        audio_lengths.append(len(wav))
        filepaths.append(s["audio_filepath"])

    max_len = max(audio_lengths)
    B = len(audio_list)
    padded_audio = torch.zeros((B, max_len), dtype=torch.float32)

    for i, wav in enumerate(audio_list):
        padded_audio[i, : len(wav)] = wav

    return {
        "audio": padded_audio,
        "audio_lengths": torch.tensor(audio_lengths, dtype=torch.long),
        "filepaths": filepaths,
    }


@hydra_runner(config_path="conf", config_name="duplex_eartts")
def inference(cfg):
    OmegaConf.resolve(cfg)

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        target_device = torch.device(f"cuda:{local_rank}")
    else:
        target_device = torch.device("cpu")

    # 1. Load the Generative Codec Model via PyTorch Lightning ckpt
    if cfg.get("checkpoint_path", None):
        print(f"Loading Model from: {cfg.checkpoint_path}")
        model = GenerativeCodecEARTTS.load_from_checkpoint(
            cfg.checkpoint_path,
            cfg=OmegaConf.to_container(cfg, resolve=True),
            map_location=target_device  
        ).eval()
        model = model.to(target_device)
    else:
        raise ValueError("For evaluation, you must provide `++checkpoint_path`.")

    target_dtype = getattr(torch, cfg.get("inference_dtype", "float32"))
    if target_dtype != torch.float32:
        model.to(dtype=target_dtype)

    # 2. Setup Metrics & Pseudo-Labeler
    eval_name = "reconstruction_eval"
    intelligibility = Intelligibility(cfg.get("scoring_asr", "stt_en_fastconformer_transducer_large"), reuse_asr_hyps=False).reset()
    secs_metric = SECS(cfg.get("scoring_speaker", "titanet_large")).reset()
    
    # Independent ASR Model for generating ground-truth pseudo-labels from the original audio
    print(f"Loading ASR Model for Pseudo-Labels: {cfg.get('scoring_asr', 'stt_en_fastconformer_transducer_large')}")
    asr_pseudo_labeler = EncDecRNNTBPEModel.from_pretrained(cfg.get("scoring_asr", "stt_en_fastconformer_transducer_large")).to(target_device).eval()

    # 3. Setup Dataset and DataLoader
    if not cfg.get("audio_dir", None):
        raise ValueError("You must provide `++audio_dir=/path/to/audio`.")
    
    eval_dataset = AudioDirectoryDataset(cfg.audio_dir)
    collate_fn = partial(collate_audio_only, sample_rate=model.target_sample_rate)

    dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.get("batch_size", 4),
        collate_fn=collate_fn,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )

    prompt_duration = cfg.get("prompt_duration", 3.0)
    max_batches = cfg.get("max_batches", -1)
    
    print(f"Starting evaluation. Total batches: {len(dataloader)}")
    if max_batches > 0:
        print(f"DEBUG MODE: Stopping early after {max_batches} batches.")

    # 4. Evaluation Loop
    for batch_id, inputs in enumerate(dataloader):
        if max_batches > 0 and batch_id >= max_batches:
            print(f"\nReached max_batches={max_batches}. Halting evaluation early.")
            break
            
        audio = inputs["audio"].to(model.device)
        audio_lengths = inputs["audio_lengths"].to(model.device)
        B = audio.size(0)

        use_autocast = target_dtype != torch.float32
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=target_dtype) if use_autocast else nullcontext()
        
        with torch.no_grad(), autocast_ctx:
            # --- A. PREPARE THE PROMPT ---
            prompt_samples = int(prompt_duration * model.target_sample_rate)
            
            # Ensure we don't consume the entire audio as a prompt
            max_prompt = (audio_lengths // 2).clamp_min(1)
            actual_prompt_samples = min(prompt_samples, max_prompt.min().item())

            model.set_init_inputs(
                speaker_audio=audio,
                speaker_audio_lens=audio_lengths,
                system_prompt=None,
                speaker_name=None
            )
            init_inputs = model.get_init_inputs(B=B)

            # --- B. PERCEPTION & QUANTIZATION BOTTLENECK ---
            asr_sr = model.cfg.get("asr_sample_rate", 16000)
            target_audio_asr_sr = resample(audio, model.target_sample_rate, asr_sr)
            target_audio_lens_asr_sr = (audio_lengths / model.target_sample_rate * asr_sr).to(torch.long)
            # During evaluation, treat the entire padded audio as a valid sequence.
            target_audio_lens_asr_sr = torch.full(
                (target_audio_asr_sr.shape[0],),
                target_audio_asr_sr.shape[1],
                dtype=torch.long,
                device=target_audio_asr_sr.device,
            )
            encoded, encoded_len = model.perception(
                input_signal=target_audio_asr_sr, input_signal_length=target_audio_lens_asr_sr
            )

            if model.cfg.get("use_pretrained_quantizer", False):
                encoded = model.quantizer_projection(encoded)
            else:
                if getattr(model, "use_fsq", False):
                    z = model.quantizer_bottleneck(encoded)
                    with fp32_precision():
                        if not model.cfg.get("skip_fsq", False):
                            z_q, _ = model.vector_quantizer(inputs=z.transpose(1, 2), input_len=encoded_len)
                        else:
                            z_q = z.transpose(1, 2)
                    z_q = z_q.transpose(1, 2).to(z.dtype)
                    encoded = model.quantizer_projection(z_q)

            # --- C. RECONSTRUCTION INFERENCE ---
            recon_audio, recon_lens = model.offline_inference(
                next_asr_embs=encoded,
                init_inputs=init_inputs,
                guidance_enabled=model.cfg.get("inference_guidance_enabled", True)
            )

            # return len is the whole audio reset it consideting 
            recon_lens = audio_lengths.clone()
            # add delay frames if used
            delay_frames = model.cfg.get("num_delay_speech_tokens", 0) + 4 # give more 0.32s to avoid cuts given it is a generative model
            if delay_frames:
                samples_per_frame_out = int(model.target_sample_rate * model.frame_length)
                delay_samples_out = int(delay_frames * samples_per_frame_out)
                recon_lens = recon_lens + delay_samples_out

            # Cap the recon_lens so it doesn't exceed the actual generated tensor size
            max_generated_samples = recon_audio.shape[1]
            recon_lens = torch.clamp(recon_lens, max=max_generated_samples)

            # --- D. METRIC PREPARATION ---
            # Isolate the portion of the original audio we asked the model to reconstruct
            target_audio_no_prompt = audio
            target_audio_no_prompt_lens = audio_lengths

            recon_audio = recon_audio.float()
            target_audio_no_prompt = target_audio_no_prompt.float()

            with fp32_precision():
                # Resample both to 16kHz for metrics
                recon_audio_16k = resample(recon_audio, model.target_sample_rate, 16000)
                recon_lens_16k = (recon_lens / model.target_sample_rate * 16000).to(torch.long)

                target_audio_16k = resample(target_audio_no_prompt, model.target_sample_rate, 16000)
                target_lens_16k = (target_audio_no_prompt_lens / model.target_sample_rate * 16000).to(torch.long)

                # Generate Pseudo-Labels from the Ground Truth target audio
                target_asr_texts = asr_pseudo_labeler.transcribe(
                    [wav[:alen].cpu().numpy() for wav, alen in zip(target_audio_16k, target_lens_16k)],
                    batch_size=B,
                    verbose=False,
                )
                pseudo_texts = [asr_hyp.text for asr_hyp in target_asr_texts]

                # --- E. UPDATE METRICS ---
                asr_hyps = intelligibility.update(
                    name=eval_name,
                    refs=pseudo_texts,
                    pred_audio=recon_audio_16k,
                    pred_audio_lens=recon_lens_16k,
                )

                secs_metric.update(
                    name=eval_name,
                    target_audio=target_audio_16k,
                    target_audio_lens=target_lens_16k,
                    pred_audio=recon_audio_16k,
                    pred_audio_lens=recon_lens_16k,
                )

        # --- F. SAVE SAMPLES ---
        out_dir = cfg.get("out_dir", "./reconstructed_samples")
        os.makedirs(out_dir, exist_ok=True)
        recon_audio_cpu = recon_audio.detach().cpu().float()
        recon_lens_cpu = recon_lens.cpu()
        
        # Prepare GT audio for saving
        gt_audio_cpu = audio.detach().cpu().float()
        gt_lens_cpu = audio_lengths.cpu()

        for i in range(B):
            target_path = inputs["filepaths"][i]
            base_name = os.path.basename(target_path)
            base_name_no_ext = os.path.splitext(base_name)[0]
            
            # Save Reconstructed Audio
            wav_recon = recon_audio_cpu[i, : recon_lens_cpu[i]].numpy()
            out_path_recon = os.path.join(out_dir, f"{base_name_no_ext}_recon.wav")
            sf.write(out_path_recon, wav_recon, samplerate=model.target_sample_rate)
            
            # Save Ground Truth Audio
            wav_gt = gt_audio_cpu[i, : gt_lens_cpu[i]].numpy()
            out_path_gt = os.path.join(out_dir, f"{base_name_no_ext}_gt.wav")
            sf.write(out_path_gt, wav_gt, samplerate=model.target_sample_rate)
        
        print(f"Processed Batch {batch_id+1}/{len(dataloader)}")

    # --- FINAL METRICS ---
    print("\n" + "="*50)
    print("--- GENERATIVE CODEC EVALUATION METRICS ---")
    
    # Intelligibility (WER/CER)
    cer_wer = intelligibility.compute()
    for k, m in cer_wer.items():
        print(f"Intelligibility - {k}: {m}")

    # SECS (Speaker Similarity)
    secs_scores = secs_metric.compute()
    for k, m in secs_scores.items():
        print(f"SECS - {k}: {m}")
        
    print("="*50)
    print(f"Reconstructed audio saved to: {cfg.get('out_dir')}")

if __name__ == "__main__":
    inference()
