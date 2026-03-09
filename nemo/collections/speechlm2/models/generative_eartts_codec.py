import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from nemo.collections.asr.models import ASRModel
from nemo.collections.speechlm2.parts.pretrained import setup_speech_encoder
from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.optim_setup import is_frozen
from nemo.collections.tts.modules.audio_codec_modules import FiniteScalarQuantizer
from nemo.collections.speechlm2.models.duplex_ear_tts import RVQEARTTSModel, DuplexEARTTS, setup_audio_codec, replace_control_speech_codes, ensures_target_precision
from types import SimpleNamespace


class GenerativeCodecRVQEARTTSModel(RVQEARTTSModel):
    """
    Overrides RVQEARTTSModel to ignore textual inputs and prioritize 
    the ASR/FSQ embedding for generative codec modeling.
    """
    def __init__(self, config, tokenizer=None):
        super().__init__(config, tokenizer)
        
        # Completely remove unused textual modules from the PyTorch registry
        # This saves VRAM and prevents FSDP gradient synchronization crashes
        if hasattr(self, 'embed_subword') and self.embed_subword is not None:
            del self.embed_subword
            self.embed_subword = None
            
        if hasattr(self, 'embed_context') and self.embed_context is not None:
            del self.embed_context
            self.embed_context = None

    def forward(self, *args, **kwargs):
        # Explicitly nullify text-based inputs to force the model 
        # to rely entirely on `asr_speech_tokens_emb`
        kwargs['subword_ids'] = None
        kwargs['subword_mask'] = None
        kwargs['context_hidden_state'] = None
        
        return super().forward(*args, **kwargs)


class GenerativeCodecEARTTS(DuplexEARTTS):
    """
    Inherits from DuplexEARTTS to instantiate an ASR encoder and FSQ quantizer,
    bypassing text conditioning for a pure generative codec approach.
    """
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        
        # Replace base TTS model with our text-ignoring class
        self.tts_model = GenerativeCodecRVQEARTTSModel(
            DictConfig(self.cfg.tts_config), 
            tokenizer=self.tokenizer
        )
        # Replace codec and also load rvq embeddings
        setup_audio_codec(self)

        # Temporarily mock self.llm so setup_speech_encoder works here
        self.llm = SimpleNamespace(
            config=SimpleNamespace(hidden_size=self.tts_model.hidden_size)
        )
        # Setup the Speech Encoder (ASR model perception)
        setup_speech_encoder(self, pretrained_weights=True)

        # Setup the FSQ Quantizer
        self.use_fsq = self.cfg.get("fsq_quantizer_levels", None) is not None
        if self.use_fsq:
            bottleneck_dim = len(self.cfg.fsq_quantizer_levels)
            hidden_size = self.tts_model.hidden_size 

            self.quantizer_bottleneck = nn.Linear(hidden_size, bottleneck_dim)
            self.vector_quantizer = FiniteScalarQuantizer(self.cfg.fsq_quantizer_levels)
            self.quantizer_projection = nn.Linear(bottleneck_dim, hidden_size)

        self.frame_length = cfg["data"]["frame_length"]

    def prepare_inputs(self, batch: dict):
        inputs = super().prepare_inputs(batch)
        
        # Prepare target audio for the ASR encoder
        target_audio_asr_sr = resample(
            batch["target_audio"], 
            self.target_sample_rate, 
            self.cfg.get("asr_sample_rate", 16000)
        )
        target_audio_lens_asr_sr = (
            batch["target_audio_lens"] / self.target_sample_rate * self.cfg.get("asr_sample_rate", 16000)
        ).to(torch.long)

        
        delay_frames = self.cfg.get("num_delay_speech_tokens", 0)
        if delay_frames > 0:
            # Calculate how many audio samples correspond to the delay frames at the ASR sample rate
            samples_per_frame_asr = int(self.cfg.get("asr_sample_rate", 16000) * self.frame_length)
            delay_samples_asr = int(delay_frames * samples_per_frame_asr)
            
            if target_audio_asr_sr.shape[1] > delay_samples_asr:
                # Remove the delay samples from the start
                shifted_audio = target_audio_asr_sr[:, delay_samples_asr:]
                
                # Pad with zeros at the end to maintain tensor shape
                zeros_pad = torch.zeros(
                    target_audio_asr_sr.size(0), delay_samples_asr, 
                    device=target_audio_asr_sr.device, dtype=target_audio_asr_sr.dtype
                )
                target_audio_asr_sr = torch.cat([shifted_audio, zeros_pad], dim=1)

        # Generate the ASR embedding
        encoded, encoded_len = self.perception(
            input_signal=target_audio_asr_sr, 
            input_signal_length=target_audio_lens_asr_sr
        )

        # Apply FSQ Quantization
        if self.use_fsq:
            z = self.quantizer_bottleneck(encoded)
            with fp32_precision():
                z_q, _ = self.vector_quantizer(inputs=z.transpose(1, 2), input_len=encoded_len)
            z_q = z_q.transpose(1, 2).to(z.dtype)
            encoded = self.quantizer_projection(z_q)

        # Align sequence lengths to the target codes
        target_len = inputs["code"].shape[1]
        if encoded.shape[1] < target_len:
            encoded = F.pad(encoded, (0, 0, 0, target_len - encoded.shape[1]))
        elif encoded.shape[1] > target_len:
            encoded = encoded[:, :target_len, :]

        inputs["asr_speech_tokens_emb"] = encoded

        # Wipe text inputs
        inputs["subword_ids"] = None
        inputs["subword_mask"] = None
        inputs["context_hidden_state"] = None
        
        return inputs

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.tts_model,):
            if is_frozen(m):
                m.eval()

        inputs = self.prepare_inputs(batch)

        tts_output = self.tts_model(
            code=inputs["code"],
            audio_mask=inputs["audio_mask"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            context_hidden_state=None,
            subword_ids=None,
            subword_mask=None,
            non_prompt_mask=inputs["non_prompt_mask"],
            dataset_type=batch.get("dataset_type", None),
            tiled_prompt_audio_codes=inputs["tiled_prompt_audio_codes"],
            tiled_prompt_subword_ids=inputs["tiled_prompt_subword_ids"],
            tiled_prompt_subword_mask=inputs["tiled_prompt_subword_mask"],
            asr_speech_tokens_emb=inputs["asr_speech_tokens_emb"], 
        )
        
        loss_dict = {"lm_loss": tts_output.lm_loss, "c_loss": tts_output.c_loss, "k_loss": tts_output.k_loss}
        loss = sum(loss_dict.values())
        num_frames = inputs["output_lens"].sum()
        B, T = inputs["code"].shape[:2]
        
        ans = {
            "loss": loss,
            "learning_rate": torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0),
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),
            "padding_ratio": num_frames / (B * T),
            **loss_dict,
        }
        self.log_dict(ans, on_step=True)
        return ans

    def get_teacher_force_inference_audio(self, batch, guidance_enabled=True):
        inputs = self.prepare_inputs(batch)

        tts_output = self.tts_model(
            code=inputs["code"],
            audio_mask=inputs["audio_mask"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            context_hidden_state=None,
            subword_ids=None,
            subword_mask=None,
            non_prompt_mask=inputs["non_prompt_mask"],
            generation_config=self._get_generation_config(guidance_enabled=guidance_enabled),
            teacher_forcing_inference=True,
            guidance_enabled=guidance_enabled,
            asr_speech_tokens_emb=inputs["asr_speech_tokens_emb"], 
        )
        tf_audio_codes_pred = tts_output["codes"].squeeze(2)

        tf_audio_codes_pred = replace_control_speech_codes(
            tf_audio_codes_pred, self._control_codes, self.codec_silence_tokens
        )
        with ensures_target_precision(self.audio_codec_run_dtype), torch.no_grad():
            audio_pred, audio_len = self.audio_codec.decode(tf_audio_codes_pred, inputs["output_lens"])

        return audio_pred.squeeze(1), audio_len

    @torch.inference_mode()
    def infer_codes_one_step(
        self,
        current_asr_emb,
        current_subword_mask,
        prev_audio_tokens,
        past_key_values,
        guidance_enabled=True,
        generation_config=None,
        ignore_eos_flag_stop=True,
        tiled_prompt_audio_codes=None,
        tiled_prompt_subword_ids=None,
        tiled_prompt_subword_mask=None,
    ):
        inputs = {
            "code": prev_audio_tokens,
            "context_hidden_state": None,
            "subword_ids": None,
            "subword_mask": None,
            "past_key_values": past_key_values,
            "use_cache": True,
            "guidance_enabled": guidance_enabled,
            "generation_config": generation_config,
            "ignore_eos_flag_stop": ignore_eos_flag_stop,
            "tiled_prompt_audio_codes": tiled_prompt_audio_codes,
            "tiled_prompt_subword_ids": tiled_prompt_subword_ids,
            "tiled_prompt_subword_mask": tiled_prompt_subword_mask,
            "asr_speech_tokens_emb": current_asr_emb, 
        }

        outputs = self.tts_model(**inputs)
        return outputs["codes"], outputs["past_key_values"]

    @torch.inference_mode()
    def offline_inference(
        self,
        next_asr_embs: torch.Tensor,
        init_inputs: dict,
        task: str = "",
        guidance_enabled: bool = True,
        generation_config: dict = None,
        incremental_audio_decoding: bool = False,
    ) -> dict[str, torch.Tensor]:
        
        B = next_asr_embs.size(0)
        
        if self.cfg.tts_config.get("use_tiled_prompt_channel", False):
            base_prompt_audio_codes = init_inputs.pop("base_prompt_audio_codes")
            base_prompt_subword_ids = init_inputs.pop("base_prompt_subword_ids")
            p_lens = init_inputs.pop("p_lens")
            safe_p_lens = p_lens.clamp_min(1)

        if generation_config is None:
            generation_config = self._get_generation_config(guidance_enabled)

        init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": guidance_enabled})

        # Warmup the model with prompt inputs
        outputs = self.tts_model(**init_inputs)

        if self.cfg.get("inference_skip_first_code_prediction_on_init", True):
            code = init_inputs["code"][:, -1:]
        else:
            code, _, _ = self.tts_model.generate_step(outputs.hidden_states[:, -1:], **generation_config)

        past_key_values = outputs["past_key_values"]
        max_steps = next_asr_embs.size(1)
        
        gen_audio_codes = torch.zeros(
            B, max_steps, self.tts_model.config.num_quantizers, device=self.device, dtype=torch.long
        )

        audio_pred = None
        audio_pred_len = torch.zeros(B, device=self.device, dtype=torch.long)

        for i in range(max_steps):
            # Extract current step ASR embedding [B, 1, H]
            current_asr_emb = next_asr_embs[:, i].unsqueeze(1)
            
            if self.cfg.tts_config.get("use_tiled_prompt_channel", False):
                t_abs = p_lens + i
                mod_idx = (t_abs % safe_p_lens).unsqueeze(1)
                step_tiled_text = torch.gather(base_prompt_subword_ids, 1, mod_idx)
                C = base_prompt_audio_codes.shape[-1]
                gather_indices_audio = mod_idx.unsqueeze(-1).expand(-1, -1, C)
                step_tiled_audio = torch.gather(base_prompt_audio_codes, 1, gather_indices_audio)
                step_tiled_mask = torch.ones_like(step_tiled_text, dtype=torch.bool)
            else:
                step_tiled_audio = None
                step_tiled_text = None
                step_tiled_mask = None

            current_subword_mask = torch.ones(B, 1, device=self.device, dtype=torch.bool)

            code, past_key_values = self.infer_codes_one_step(
                current_asr_emb=current_asr_emb,
                current_subword_mask=current_subword_mask,
                prev_audio_tokens=code,
                past_key_values=past_key_values,
                guidance_enabled=guidance_enabled,
                generation_config=generation_config,
                ignore_eos_flag_stop=True,
                tiled_prompt_audio_codes=step_tiled_audio,
                tiled_prompt_subword_ids=step_tiled_text,
                tiled_prompt_subword_mask=step_tiled_mask,
            )

            gen_audio_codes[:, i] = code.squeeze(1)

            if incremental_audio_decoding:
                audio_pred_i, audio_pred_i_len = self.decode_one_audio_step(
                    gen_audio_codes[:, : i + 1],
                    number_prev_tokens=self.cfg.get("inference_codec_decoding_prev_tokens_number", None),
                )
                if audio_pred is None:
                    audio_pred = audio_pred_i
                else:
                    audio_pred = torch.cat([audio_pred, audio_pred_i], dim=1)
                audio_pred_len += audio_pred_i_len

        if not incremental_audio_decoding:
            gen_audio_codes_lens = torch.tensor([gen_audio_codes.shape[1]] * gen_audio_codes.shape[0]).to(self.device)
            gen_audio_codes = replace_control_speech_codes(
                gen_audio_codes, self._control_codes, self.codec_silence_tokens
            )
            with ensures_target_precision(self.audio_codec_run_dtype), torch.no_grad():
                audio_pred, audio_pred_len = self.audio_codec.decode(gen_audio_codes, gen_audio_codes_lens)

        return audio_pred.squeeze(1), audio_pred_len

    def set_init_inputs(self, speaker_audio=None, speaker_audio_lens=None, system_prompt=None, speaker_name=None):
        # 1. Let the parent handle the complex setup, codec encoding, and caching of base tensors
        init_inputs = super().set_init_inputs(speaker_audio, speaker_audio_lens, system_prompt, speaker_name)
        
        # 2. Recreate the exact target_audio that the parent just built so we can pass it to the ASR encoder
        with fp32_precision():
            prompt_audio_size = int(
                ((self.data_cfg.audio_prompt_duration * self.target_sample_rate) // self.target_samples_per_frame)
                * self.target_samples_per_frame
            )
            
            if speaker_name is not None:
                speaker_audio = torch.zeros((1, prompt_audio_size), device=self.device, dtype=torch.float32)
                speaker_audio_lens = torch.LongTensor([speaker_audio.shape[1]]).to(self.device)

            B, T = speaker_audio.shape
            prompt_audio = torch.zeros(B, prompt_audio_size, device=self.device, dtype=speaker_audio.dtype)
            
            for b in range(B):
                valid_len = min(speaker_audio_lens[b].item(), T)
                if valid_len <= 0:
                    continue
                valid_segment = speaker_audio[b, :valid_len]
                if valid_len >= prompt_audio_size:
                    prompt_audio[b] = valid_segment[:prompt_audio_size]
                else:
                    repeat_factor = (prompt_audio_size + valid_len - 1) // valid_len
                    expanded = valid_segment.repeat(repeat_factor)
                    prompt_audio[b] = expanded[:prompt_audio_size]
                    
            prompt_audio[:, -int(self.target_samples_per_frame * 2) :] = 0

            # Recreate the text pad sizing
            if system_prompt is not None and self.cfg.get("use_system_prompt", None) and system_prompt != "":
                text_prompt = torch.as_tensor(
                    [self.tokenizer.bos] + self.tokenizer.text_to_ids(system_prompt) + [self.tokenizer.eos],
                    dtype=torch.long, device=self.device,
                )
            else:
                text_prompt = torch.tensor([self.tokenizer.eos], dtype=torch.long, device=self.device)

            pad_size = text_prompt.size(-1) * self.target_samples_per_frame
            pad_audio = torch.zeros(pad_size, device=prompt_audio.device, dtype=prompt_audio.dtype).unsqueeze(0).repeat(B, 1)
            
            target_audio = torch.cat([pad_audio, prompt_audio], dim=1)
            target_audio_len = torch.tensor([target_audio.size(-1)] * B, dtype=torch.long, device=self.device)

        # 3. Get ASR embeddings for this recreated target_audio
        asr_sr = self.cfg.get("asr_sample_rate", 16000)
        target_audio_asr_sr = resample(target_audio, self.target_sample_rate, asr_sr)
        target_audio_lens_asr_sr = (target_audio_len / self.target_sample_rate * asr_sr).to(torch.long)

        # Apply the exact same lookahead shift we designed earlier!
        delay_frames = self.cfg.get("num_delay_speech_tokens", 0)
        if delay_frames > 0:
            samples_per_frame_asr = int(self.cfg.get("asr_sample_rate", 16000) * self.frame_length)
            delay_samples_asr = int(delay_frames * samples_per_frame_asr)
            if target_audio_asr_sr.shape[1] > delay_samples_asr:
                shifted_audio = target_audio_asr_sr[:, delay_samples_asr:]
                zeros_pad = torch.zeros(target_audio_asr_sr.size(0), delay_samples_asr, device=target_audio_asr_sr.device, dtype=target_audio_asr_sr.dtype)
                target_audio_asr_sr = torch.cat([shifted_audio, zeros_pad], dim=1)

        encoded, encoded_len = self.perception(input_signal=target_audio_asr_sr, input_signal_length=target_audio_lens_asr_sr)

        if self.use_fsq:
            z = self.quantizer_bottleneck(encoded)
            with fp32_precision():
                z_q, _ = self.vector_quantizer(inputs=z.transpose(1, 2), input_len=encoded_len)
            z_q = z_q.transpose(1, 2).to(z.dtype)
            encoded = self.quantizer_projection(z_q)

        # The parent init_inputs drops the last frame (`[:, :-1]`), so target_len is code length + 1
        target_len = init_inputs["code"].shape[1] + 1
        if encoded.shape[1] < target_len:
            encoded = F.pad(encoded, (0, 0, 0, target_len - encoded.shape[1]))
        elif encoded.shape[1] > target_len:
            encoded = encoded[:, :target_len, :]

        # 4. Inject into init_inputs (applying the same [:, :-1] slice)
        init_inputs["asr_speech_tokens_emb"] = encoded[:, :-1]

        # 5. Nullify text inputs securely
        init_inputs["subword_ids"] = None
        init_inputs["subword_mask"] = None
        init_inputs["context_hidden_state"] = None

        # 6. Update the cache so get_init_inputs finds them later
        self._init_input_cache["asr_speech_tokens_emb"] = init_inputs["asr_speech_tokens_emb"].detach().clone()
        self._init_input_cache["subword_ids"] = None
        self._init_input_cache["subword_mask"] = None
        self._init_input_cache["context_hidden_state"] = None

        return init_inputs

    def get_init_inputs(self, B: int, init_inputs_names=None):
        if init_inputs_names is None:
            init_inputs_names = [
                "code",
                "audio_mask",
                "non_prompt_mask",
            ]

        # Ensure our custom embedding is requested from the cache
        if "asr_speech_tokens_emb" not in init_inputs_names:
            init_inputs_names.append("asr_speech_tokens_emb")

        init_inputs = super().get_init_inputs(B, init_inputs_names)
        
        # Ensure textual inputs are explicitly None so the generative codec relies strictly on ASR
        init_inputs["subword_ids"] = None
        init_inputs["subword_mask"] = None
        init_inputs["context_hidden_state"] = None

        return init_inputs

    @torch.inference_mode()
    def run_evaluation_one_batch(self, name, dataset_batch, use_dataloader_init=False):
        results = {}
        inputs = self.prepare_inputs(dataset_batch)
        asr_emb = inputs["asr_speech_tokens_emb"]

        results["audio_tf"], results["audio_tf_len"] = self.get_teacher_force_inference_audio(dataset_batch)

        if use_dataloader_init:
            init_inputs = {
                "code": inputs["code"],
                "audio_mask": inputs["audio_mask"],
                "non_prompt_mask": inputs["non_prompt_mask"],
                "asr_speech_tokens_emb": asr_emb,
            }
            for key in init_inputs:
                if init_inputs[key] is not None:
                    init_inputs[key] = torch.stack(
                        [init_inputs[key][i, :plen] for i, plen in enumerate(dataset_batch["prompt_lens"])]
                    )
        else:
            sp = dataset_batch.get("system_prompts_raw")
            system_prompt = sp[0] if sp else None
            self.set_init_inputs(
                speaker_audio=dataset_batch["audio_prompt"],
                speaker_audio_lens=dataset_batch["audio_prompt_lens"],
                system_prompt=system_prompt,
            )
            init_inputs = self.get_init_inputs(B=inputs["code"].size(0))
            
            # Inject corresponding prompt portion of the ASR embedding into init_inputs
            target_len = init_inputs["code"].shape[1]
            init_inputs["asr_speech_tokens_emb"] = torch.stack([
                asr_emb[i, :target_len] for i in range(asr_emb.size(0))
            ])

        # Slice remaining ASR embeddings to represent the 'next' sequences to generate
        next_asr_embs = torch.stack([
            asr_emb[i, plen:] for i, plen in enumerate(dataset_batch["prompt_lens"])
        ])

        results["audio"], results["audio_len"] = self.offline_inference(
            next_asr_embs=next_asr_embs,
            init_inputs=init_inputs,
            task=dataset_batch["task"][0],
        )

        dataset_batch["source_audio"] = dataset_batch["source_audio"][
            :, -int(next_asr_embs.size(1) * self.source_samples_per_frame) :
        ]

        results["audio_tf"] = results["audio_tf"][:, -int(next_asr_embs.size(1) * self.target_samples_per_frame) :]
        
        target_audio_no_prompt = dataset_batch["target_audio"][
            :, -int(next_asr_embs.size(1) * self.target_samples_per_frame) :
        ]
        target_audio_no_prompt_lens = dataset_batch["target_audio_lens"] - (
            torch.tensor(
                dataset_batch["prompt_lens"], dtype=torch.long, device=dataset_batch["target_audio_lens"].device
            ) * self.target_samples_per_frame
        )

        with fp32_precision():
            metric_audio_pred = results["audio"]
            metric_audio_pred_lens = results["audio_len"]

            metric_audio_pred = resample(metric_audio_pred, self.target_sample_rate, 16000)
            metric_audio_pred_lens = (metric_audio_pred_lens / self.target_sample_rate * 16000).to(torch.long)
            target_audio_no_prompt_16khz = resample(target_audio_no_prompt, self.target_sample_rate, 16000)
            target_audio_no_prompt_lens_16khz = (target_audio_no_prompt_lens / self.target_sample_rate * 16000).to(torch.long)
            
            if self.cfg.get("use_GT_transcriptions_for_metrics", True):
                target_asr_texts = self.asr_bleu.asr.transcribe(
                    [
                        audio[:alen]
                        for audio, alen in zip(target_audio_no_prompt_16khz, target_audio_no_prompt_lens_16khz)
                    ],
                    batch_size=target_audio_no_prompt_16khz.shape[0],
                    verbose=False,
                )
                metric_text = [asr_hyp.text for asr_hyp in target_asr_texts]
            else:
                metric_text = dataset_batch["target_texts"]

            asr_hyps = self.asr_bleu.update(
                name=name,
                refs=metric_text,
                pred_audio=metric_audio_pred,
                pred_audio_lens=metric_audio_pred_lens,
            )

            self.intelligibility.update(
                name=name,
                refs=metric_text,
                pred_audio=metric_audio_pred,
                pred_audio_lens=metric_audio_pred_lens,
                asr_hyps=asr_hyps,
            )

            self.intelligibility.update(
                name=name + "_gt",
                refs=dataset_batch["target_texts"],
                pred_audio=target_audio_no_prompt_16khz,
                pred_audio_lens=target_audio_no_prompt_lens_16khz,
                asr_hyps=(metric_text if self.cfg.get("use_GT_transcriptions_for_metrics", True) else None), 
            )

            self.secs.update(
                name=name,
                target_audio=resample(dataset_batch["target_audio"], self.target_sample_rate, 16000),
                target_audio_lens=(dataset_batch["target_audio_lens"] / self.target_sample_rate * 16000).to(torch.long),
                pred_audio=resample(results["audio"], self.target_sample_rate, 16000),
                pred_audio_lens=(results["audio_len"] / self.target_sample_rate * 16000).to(torch.long),
            )

            # NOTE: EOU labels are skipped since we dropped the textual representation entirely
            # eou_labels = ...

            self.results_logger.update(
                name=name,
                refs=dataset_batch["target_texts"],
                hyps=metric_text,
                asr_hyps=asr_hyps,
                samples_id=dataset_batch['sample_id'],
                pred_audio=results["audio"].float(),
                pred_audio_tf=results["audio_tf"].float(),
                pre_audio_trimmed=None,
                reference_audio=dataset_batch["audio_prompt"].float(),
                target_audio=target_audio_no_prompt.float(),
                pred_audio_sr=self.target_sample_rate,
                user_audio=dataset_batch["source_audio"].float(),
                user_audio_sr=self.source_sample_rate,
                eou_pred=None, 
                fps=self.target_fps,
                results=results if self.cfg.get("dump_tokens_text", False) else None,
                tokenizer=self.tokenizer,
            )

