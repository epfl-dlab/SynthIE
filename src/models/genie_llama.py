import os
from typing import List, Any

import hydra
import torch
import transformers
import pandas as pd
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config, LlamaForCausalLM, LlamaConfig
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

import src.utils as utils
from src.metrics import TSF1, TSPrecision, TSRecall
from src.models.collators import GenericCollator
from src.utils import general_helpers, evaluation_helpers
from src.utils.training_utils import label_smoothed_nll_loss


log = utils.get_pylogger(__name__)


class GenIELlamaPL(LightningModule):
    def __init__(
        self,
        hparams_overrides=None,
        hf_config_overrides=None,
        from_pretrained=False,
        generate_predictions_in_sanity_check=False,
        constraint_module=None,
        **kwargs,
    ):
        super().__init__()
        self._generate_predictions_in_sanity_check = generate_predictions_in_sanity_check

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "generate_predictions_in_sanity_check",
                "hparams_overrides",
                "hf_config_overrides",
                "datamodule",
                "collator",
                "constraint_module",
            ],
        )

        if hparams_overrides is not None:
            self._override_checkpoint_hparams(hparams_overrides)

        # ~~~ Load the tokenizer ~~~
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model_name_or_path)
        # set padding token to pad_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token # llama doesn't have a pad token, so we use eos instead
        # using the eos token as the pad token is recommended in the error message:
        # Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)`

        # ~~~ Get the HF config ~~~
        hf_config = LlamaConfig.from_pretrained(self.hparams.pretrained_model_name_or_path)
        # Override the HF config with values from the checkpoint (if loading from checkpoint)
        if self.hparams.get("hf_config", None):
            hf_config.update(self.hparams.hf_config.to_dict())
        # Override HF config parameters (if it applies)
        if hf_config_overrides is not None:
            hf_config.update(hf_config_overrides)
        # Update the hparams with the updated config
        self.hparams.hf_config = hf_config

        # ~~~ Load the model ~~~
        if from_pretrained:
            self.model = LlamaForCausalLM.from_pretrained(
                self.hparams.pretrained_model_name_or_path, config=self.hparams.hf_config
            )
        else:
            self.model = LlamaForCausalLM(config=self.hparams.hf_config)

        log.info("HF model config:")
        log.info(self.hparams.hf_config)

        # ~~~ Set collator ~~~
        self.collator = kwargs.get("collator", None)
        if self.collator is None:
            self.collator = self._get_default_collator()
        else:
            self.collator.set_tokenizer(self.tokenizer)

        # ~~~ Initialize metrics ~~~
        self.ts_precision = TSPrecision()
        self.ts_recall = TSRecall()
        self.ts_f1 = TSF1()

        # ~~~ Constraint generation ~~~
        if constraint_module is not None:
            log.info("Running inference with CONSTRAINED decoding")
            self.constraint_module = hydra.utils.instantiate(constraint_module, model=self)
        else:
            log.info("Running UNCONSTRAINED inference.")
            self.constraint_module = None

        # ~~~ Inference ~~~
        linearization_class_id = self.hparams.get("linearization_class_id", None)
        log.info(f"Linearization class ID: {linearization_class_id}")

        self.linearization_class = utils.get_linearization_class(linearization_class_id)
        self.output_dir = None

    def _override_checkpoint_hparams(self, hparams_overrides: dict):
        """
        Overrides the hyperparameters of a checkpoint at an arbitrary depth
        :param hparams_overrides:
        :return:
        """
        general_helpers.rec_dict_update(self.hparams, hparams_overrides)
        log.info("Some values of the original hparams were overridden")
        log.info("Hyper-parameters:")
        log.info(self.hparams)

    def _get_default_collator(self):
        return GenericCollator(tokenizer=self.tokenizer, **self.hparams.default_collator_parameters)

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None, **kwargs):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            **kwargs,
        )

        return output

    def process_batch(self, batch):
        batch["decoder_input_ids"] = self.model._shift_right(batch["tgt_input_ids"])

        return batch

    @rank_zero_only
    def _sanity_check(self, batch, batch_idx, stage, dataloader_idx=0, predictions=None):
        if batch_idx >= 2 and not self.trainer.sanity_checking:
            return

        ids = batch["id"]
        raw_input = [sample["text"] for sample in batch["raw"]]
        raw_target = [sample["target"] for sample in batch["raw"]]
        num_samples_to_show = min(2, len(ids))

        if predictions is None and self._generate_predictions_in_sanity_check:
            sample_output = self._get_predictions_for_batch(batch, raw_input)
            predictions = sample_output["grouped_decoded_outputs"]

        batch_summary = []

        for i in range(num_samples_to_show):
            dp_id = ids[i]

            num_input_tokens = batch["src_attention_mask"][i].sum()
            input_ids = batch["src_input_ids"][i][:num_input_tokens]
            input_text = raw_input[i] + f"{self.tokenizer.eos_token}"
            decoded_input_text = self.tokenizer.decode(input_ids)
            if "<unk>" not in decoded_input_text:
                if input_text != decoded_input_text:
                    log.info(
                        f"[Input text vs. Decoded input IDs -- Batch Index {batch_idx}] Input text: {str(input_text)}"
                    )
                    log.info(
                        f"[Input text vs. Decoded input IDs -- Batch Index {batch_idx}] Decoded text: {str(decoded_input_text)}"
                    )
                    # raise ValueError("Input text and decoded input text are not the same. There is some problem with the tokenizer.")

            # input_summary = f"Input text: {input_text}\nInput IDs: {input_ids}\n"

            num_target_tokens = batch["tgt_attention_mask"][i].sum()
            target_ids = batch["tgt_input_ids"][i][:num_target_tokens]
            target_text = raw_target[i] + f"{self.tokenizer.eos_token}"
            decoded_target_text = self.tokenizer.decode(target_ids)
            if "<unk>" not in decoded_target_text:
                if target_text != decoded_target_text:
                    log.info(
                        f"[Input text vs. Decoded input IDs -- Batch Index {batch_idx}] Target text: {str(target_text)}"
                    )
                    log.info(
                        f"[Input text vs. Decoded input IDs -- Batch Index {batch_idx}] Decoded text: {str(decoded_target_text)}"
                    )
                    # raise ValueError("Target text and decoded target text are not the same. There is some problem with the tokenizer.")

            # target_summary = f"Target text: {target_text}\nTarget IDs: {target_ids}\n"

            if stage in {"train", "val"}:
                assert torch.all(
                    torch.cat((torch.tensor([self.tokenizer.pad_token_id]).to(target_ids), target_ids[:-1]))
                    == batch["decoder_input_ids"][i][:num_target_tokens]
                ), "The decoder IDs are not formatted as expected."

            # log.info(f"SanityCheck_{stage}/{batch_idx}_{i}:ID: {dp_id}\n{input_summary}\n\n{target_summary}")
            summary = {
                "id": dp_id,
                "input_text": input_text,
                "input_ids": str(input_ids.cpu().tolist()),
                "target_text": target_text,
                "target_ids": str(target_ids.cpu().tolist()),
            }
            if predictions is not None:
                summary["predictions"] = predictions[i]

            if "decoder_input_ids" in batch:
                summary["decoder_input_ids"] = str(batch["decoder_input_ids"][i][:num_target_tokens].cpu().tolist())

            batch_summary.append(summary)

        df = pd.DataFrame(batch_summary)
        utils.general_helpers.log_df(
            path=f"{stage}/batch_summary/dl-{dataloader_idx}_epoch-{self.current_epoch}_global-step-{self.global_step}_batch-{batch_idx}",
            df=df,
            logger=self.logger,
        )

    def training_step(self, batch, batch_idx):
        batch = self.process_batch(batch)

        loss, nll_loss = self._compute_loss(batch)

        self.log("train/nll_loss", nll_loss.item(), on_step=True, on_epoch=False, prog_bar=True)
        self._sanity_check(batch, batch_idx, stage="train")

        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch = self.process_batch(batch)

        loss, nll_loss = self._compute_loss(batch)
        # self.log("val/nll_loss", nll_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            f"val/nll_loss_dl-{dataloader_idx}",
            nll_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        self._sanity_check(batch, batch_idx, stage="val", dataloader_idx=dataloader_idx)

        return {"val/nll_loss": nll_loss}

    def _compute_loss(self, batch):
        model_output = self(
            input_ids=batch["src_input_ids"],
            attention_mask=batch["src_attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["tgt_attention_mask"],
            use_cache=False,
        )

        logits = model_output.logits

        loss, nll_loss = label_smoothed_nll_loss(
            logits.log_softmax(dim=-1),
            batch["tgt_input_ids"],
            batch["tgt_attention_mask"],
            epsilon=self.hparams.eps,
            ignore_index=self.tokenizer.pad_token_id,
        )

        return loss, nll_loss

    def _get_predictions_for_batch(self, batch, raw_input):
        # ~~~ Prediction related ~~~
        # Generate predictions
        if self.constraint_module is None:
            sample_prefix_allowed_tokens_fn = None
        else:
            sample_prefix_allowed_tokens_fn = self.constraint_module.get_prefix_allowed_tokens_fn(batch=batch)
            assert sample_prefix_allowed_tokens_fn is not None

        hf_generation_params = self.hparams.inference["hf_generation_params"].copy()
        hf_generation_params.update(
            {
                "input_is_processed_batch": True,
                "return_generation_inputs": True,
                "return_generation_outputs": True,
                "output_scores": True,
            }
        )
        sample_output = self.sample(
            batch,
            prefix_allowed_tokens_fn=sample_prefix_allowed_tokens_fn,
            **hf_generation_params,
        )

        return sample_output

    def test_step(self, batch, batch_idx):
        raw_input = [sample["text"] for sample in batch["raw"]]
        raw_target = [sample["target"] for sample in batch["raw"]]
        ids = batch["id"]

        sample_output = self._get_predictions_for_batch(batch, raw_input)

        self._sanity_check(batch, batch_idx, stage="test", predictions=sample_output["grouped_decoded_outputs"])
        self._write_step_output(
            batch_idx=batch_idx, ids=ids, raw_input=raw_input, raw_target=raw_target, sample_output=sample_output
        )

        return_object = {
            "ids": ids,
            "inputs": raw_input,
            "targets": raw_target,
            "predictions": sample_output["grouped_decoded_outputs"],
        }
        return return_object

    def test_step_end(self, outputs: List[Any]):
        # Get the data in the format expected by the metrics
        predictions = [
            self.linearization_class.text_to_triplet_list(
                text=texts[0], verbose=self.hparams.inference.get("verbose_flag_in_convert_to_triple"), return_set=True
            )
            for texts in outputs["predictions"]
        ]
        targets = [
            self.linearization_class.text_to_triplet_list(
                text=text, verbose=self.hparams.inference.get("verbose_flag_in_convert_to_triple"), return_set=True
            )
            for text in outputs["targets"]
        ]

        # Update the metrics
        p = self.ts_precision(predictions, targets)
        r = self.ts_recall(predictions, targets)
        f1 = self.ts_f1(predictions, targets)

        # Log the loss
        self.log("test/precision_step", p, on_step=True, on_epoch=False, prog_bar=True)
        self.log("test/recall_step", r, on_step=True, on_epoch=False, prog_bar=True)
        self.log("test/f1_step", f1, on_step=True, on_epoch=False, prog_bar=True)

    def _write_step_output(
        self,
        batch_idx,
        ids,
        raw_input,
        raw_target,
        sample_output,
    ):
        # ~~~ Write prediction outputs to file ~~~
        num_return_sequences = len(sample_output["grouped_decoded_outputs"][0])
        sequences = sample_output["generation_outputs"].sequences
        assert isinstance(sequences, torch.Tensor)
        prediction_ids = general_helpers.chunk_elements(sequences.tolist(), num_return_sequences)

        # tokenizer_output = self.tokenize(raw_input, raw_target)
        # target_decoder_input_ids = tokenizer_output["decoder_input_ids"]

        prediction_outputs = {
            "id": ids,
            "input": raw_input,
            # "input_ids": sample_output["generation_inputs"]["input_ids"].tolist(),
            "target": raw_target,
            # "target_ids": target_decoder_input_ids.tolist(),
            "prediction": sample_output["grouped_decoded_outputs"],
            "prediction_ids": str(prediction_ids),
        }
        # if seeds is not None:
        #     prediction_outputs["seed"] = seeds

        prediction_outputs_path = os.path.join(
            evaluation_helpers.get_predictions_dir_path(self.output_dir),
            f"testing_output_{self.global_rank}.prediction.jsonl.gz",
        )

        prediction_outputs_summary = general_helpers.get_list_of_dicts(prediction_outputs)
        general_helpers.write_gzipped_jsonlines(prediction_outputs_path, prediction_outputs_summary, mode="a+")

        # ––––– Log a few batches during inference as a sanity check
        if batch_idx in [0, 3] and self.global_rank == 0:
            # pred_json = json.dumps(prediction_outputs_summary)
            # log.info(f"test_output/batch_{batch_idx}:\n{pred_json}")

            pred_df = pd.DataFrame(prediction_outputs)
            utils.general_helpers.log_df(path=f"test_batch_summary/batch_{batch_idx}", df=pred_df, logger=self.logger)

    def test_epoch_end(self, outputs):
        """Outputs is a list of test_step outputs"""
        # Log metrics aggregated across steps and processes (in ddp)
        self.log("test/precision", self.ts_precision.compute())
        self.log("test/recall", self.ts_recall.compute())
        self.log("test/f1", self.ts_f1.compute())

        return {
            "test/acc": self.ts_precision.compute(),
            "test/recall": self.ts_precision.compute(),
            "test/f1": self.ts_precision.compute(),
        }

    def on_test_epoch_end(self):
        if hasattr(torch.distributed, "is_initialized") and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Temporary solution to Hydra + PL + DDP issue
        # https://github.com/Lightning-AI/lightning/pull/11617#issuecomment-1245842064
        # https://github.com/ashleve/lightning-hydra-template/issues/393
        # problem should be resolved in PL version 1.8.3
        general_helpers._move_predictions_for_subprocesses(
            evaluation_helpers.get_predictions_dir_path(os.getcwd()),
            evaluation_helpers.get_predictions_dir_path(self.output_dir),
        )

        evaluation_helpers.upload_outputs_to_wandb(
            getattr(self, "hparams_to_log", {}),
            evaluation_helpers.get_predictions_dir_path(self.output_dir),
            logger=self.logger,
        )

    def configure_optimizers(self):
        # Apply weight decay to all parameters except for the biases and the weight for Layer Normalization

        decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        # Per-parameter optimization.
        # Each dict defines a parameter group and contains the list of parameters to be optimized in a key `params`
        # Other keys should match keyword arguments accepted by the optimizers and
        # will be used as optimization params for the parameter group
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.hparams.optimizer.weight_decay,
                "betas": (0.9, 0.999),
                "eps": self.hparams.optimizer.eps,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
                "betas": (0.9, 0.999),
                "eps": self.hparams.optimizer.eps,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.optimizer.lr,
            weight_decay=self.hparams.optimizer.weight_decay,
        )

        if self.hparams.scheduler.name == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.scheduler.warmup_updates,
                num_training_steps=self.hparams.scheduler.total_num_updates,
            )
        elif self.hparams.scheduler.name == "polynomial":
            scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.scheduler.warmup_updates,
                num_training_steps=self.hparams.scheduler.total_num_updates,
                lr_end=self.hparams.scheduler.lr_end,
            )
        elif self.hparams.scheduler.name is not None:
            raise ValueError("Unknown scheduler name {}".format(self.hparams.scheduler.name))

        lr_dict = {
            "scheduler": scheduler,  # scheduler instance
            "interval": "step",  # The unit of the scheduler's step size. 'step' or 'epoch
            "frequency": 1,  # corresponds to updating the learning rate after every `frequency` epoch/step
            # used by the LearningRateMonitor callback
            "name": f"LearningRateScheduler-{self.hparams.scheduler.name}",
        }

        return [optimizer], [lr_dict]

    @torch.no_grad()
    def sample(
        self,
        input_data,
        input_is_processed_batch=False,
        seed=None,
        skip_special_tokens=True,
        return_generation_outputs=False,
        return_generation_inputs=False,
        convert_to_triplets=False,
        force_free_generation=False,
        prefix_allowed_tokens_fn=None,
        **kwargs,
    ):
        """Input data is a list of strings or a processed batch (contains src_input_ids,
        and src_attention_mask as expected in training)"""
        # if the model is not in evaluation mode, set it and remember to reset it
        training = self.training
        if training:
            self.eval()

        hf_generation_params = self.hparams.inference["hf_generation_params"].copy()
        hf_generation_params.update(kwargs)
        hf_generation_params["return_dict_in_generate"] = True

        if seed is None:
            seed = self.hparams.inference.get("seed", None)
        if seed:
            transformers.trainer_utils.set_seed(seed)

        # Get input_ids and attention masks
        if not input_is_processed_batch:
            input_data = self.collator.collate_input(input_data)

        input_ids = input_data["src_input_ids"].to(self.device)
        attention_mask = input_data["src_attention_mask"].to(self.device)

        if force_free_generation:
            prefix_allowed_tokens_fn = None

        if prefix_allowed_tokens_fn is None and self.constraint_module is not None:
            prefix_allowed_tokens_fn = self.constraint_module.get_prefix_allowed_tokens_fn(batch_info=None)

        generate_kwargs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
            **hf_generation_params,
        }
        generation_outputs = self.model.generate(**generate_kwargs)

        # Returns a list of `num_sentences` decoded (textual) sequences
        num_return_sequences = hf_generation_params.get("num_return_sequences", 1)

        sequences = generation_outputs.sequences
        decoded_sequences = self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)
        if convert_to_triplets:
            decoded_triplets = [self.linearization_class.text_to_triplet_list(seq) for seq in decoded_sequences]
            grouped_decoded_sequences = general_helpers.chunk_elements(decoded_triplets, num_return_sequences)
        else:
            grouped_decoded_sequences = general_helpers.chunk_elements(decoded_sequences, num_return_sequences)

        if training:
            self.train()

        results = {"grouped_decoded_outputs": grouped_decoded_sequences}
        if return_generation_inputs:
            results["generate_kwargs"] = generate_kwargs
        if return_generation_outputs:
            results["generation_outputs"] = generation_outputs

        return results
