import pandas as pd
import pdb
import json
import os
import argparse
import torch

from transformers import Adafactor, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from transformers import LEDConfig

# Can't run bcs the original code from huggingface is not capable with the graph structure and sagpooling 
# from transformers import LEDTokenizer, LEDForConditionalGeneration
from .tokenization import LEDTokenizer
from .modeling import LEDForConditionalGeneration
from .dataloading import get_dataloader_summ

import sys

sys.path.append("../../")
from utils.metrics import rouge


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """
    This function is borrowed from fairseq.
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

    return loss, nll_loss


class HGSummarizer(pl.LightningModule):
    def __init__(self, args):
        super(HGSummarizer, self).__init__()
        self.args = args

        self.tokenizer = LEDTokenizer.from_pretrained(args.pretrained_primer)
        # self.model = LEDForConditionalGeneration.from_pretrained(args.pretrained_primer)

        config = LEDConfig.from_pretrained("allenai/led-base-16384")
        config.use_graph = True 
        self.model = LEDForConditionalGeneration(config)

        self.pad_token_id = self.tokenizer.pad_token_id
        self.use_ddp = self.args.speed_strategy == "ddp"
        self.docsep_token_id = self.tokenizer.convert_tokens_to_ids("<doc-sep>")
        # or use eos_token as sent-sep
        self.tokenizer.add_tokens(["<sent-sep>"])
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.sentsep_token_id = self.tokenizer.convert_tokens_to_ids("<sent-sep>")
        self.validation_outputs = []
        self.test_outputs = []  

    def forward(self, input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source,
                sents_positions_source,
                docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt):
        device = input_ids_source.device
        # print("encoding source", len(heterograph_source))
        decoder_input_ids = output_ids[:, :-1]
        # get the input ids and attention masks together
        global_attention_mask_source = torch.zeros_like(input_ids_source).to(device)
        # put global attention on <s> token
        global_attention_mask_source[:, 0] = 1
        # put global attention on <doc-sep> token
        global_attention_mask_source[input_ids_source == self.docsep_token_id] = 1
        # put global attention on <sent-sep> token
        global_attention_mask_source[input_ids_source == self.sentsep_token_id] = 1

        attention_mask = torch.ones_like(input_ids_source).to(device)
        attention_mask = attention_mask.type_as(input_ids_source)
        attention_mask[input_ids_source == self.pad_token_id] = 0
        # encoding source documents
        outputs_source = self.model(
            input_ids_source,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            global_attention_mask=global_attention_mask_source,
            use_cache=False,
            heterograph=heterograph_source,
            words_positions_source=words_positions_source,
            sents_positions_source=sents_positions_source,
            docs_positions_source=docs_positions_source
        )
        lm_logits = outputs_source.logits
        assert lm_logits.shape[-1] == self.model.config.vocab_size

        # print("encoding summary", input_ids_summary.size())
        # pdb.set_trace()
        # encoding summary
        # get the input ids and attention masks together
        global_attention_mask_summary = torch.zeros_like(input_ids_summary).to(device)
        # put global attention on <s> token
        global_attention_mask_summary[:, 0] = 1
        # put global attention on <sent-sep> token
        global_attention_mask_summary[input_ids_summary == self.sentsep_token_id] = 1
        outputs_summary = self.model(
            input_ids_summary.to(device),
            decoder_input_ids=None,
            global_attention_mask=global_attention_mask_summary,
            use_cache=False,
            heterograph=heterograph_tgt,
            words_positions_source=words_positions_tgt,
            sents_positions_source=sents_positions_tgt,
            docs_positions_source=None
        )

        return lm_logits, outputs_source.mgat_outputs, outputs_source.sagpooling_outputs, outputs_summary.mgat_outputs
        # return lm_logits, outputs_source.mgat_outputs, outputs_source.sagpooling_outputs, None

    def configure_optimizers(self):
        if self.args.adafactor:
            optimizer = Adafactor(
                self.parameters(),
                lr=self.args.lr,
                scale_parameter=False,
                relative_step=False,
            )
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=self.args.warmup_steps
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.total_steps,
            )
        if self.args.fix_lr:
            return optimizer
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def shared_step(self, input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source,
                    sents_positions_source,
                    docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt):
        lm_logits, mgat_outputs_source, sagpooling_ouputs, mgat_outputs_summary = self.forward(input_ids_source,
                                                                                               output_ids,
                                                                                               input_ids_summary,
                                                                                               heterograph_source,
                                                                                               words_positions_source,
                                                                                               sents_positions_source,
                                                                                               docs_positions_source,
                                                                                               heterograph_tgt,
                                                                                               words_positions_tgt,
                                                                                               sents_positions_tgt)

        # print("shared step")
        # graph similarity loss
        cos = torch.nn.CosineSimilarity(dim=1)
        graph_sim = torch.mean(cos(sagpooling_ouputs, mgat_outputs_summary))
        # coss-entropy loss
        labels = output_ids[:, 1:].clone()

        if self.args.label_smoothing == 0.0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                labels,
                self.args.label_smoothing,
                ignore_index=self.pad_token_id,
            )
        if torch.isnan(loss):
            pdb.set_trace()

        return 0.5 * loss + 0.5 * graph_sim

    def training_step(self, batch, batch_idx):
        input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source, sents_positions_source, docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt = batch
        loss = self.shared_step(input_ids_source, output_ids, input_ids_summary, heterograph_source,
                                words_positions_source,
                                sents_positions_source, docs_positions_source, heterograph_tgt, words_positions_tgt,
                                sents_positions_tgt)
        # print("training step")

        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]["lr"]
        tensorboard_logs = {
            "train_loss": loss,
            "lr": lr,
            "input_size_source": input_ids_source.numel(),
            "output_size": output_ids.numel(),
            "mem": torch.cuda.memory_allocated(loss.device) / 1024 ** 3
            if torch.cuda.is_available()
            else 0,
        }
        self.logger.log_metrics(tensorboard_logs, step=self.global_step)
        return loss

    def compute_rouge_batch(self, input_ids, gold_str, heterograph_source, words_positions_source,
                            sents_positions_source, docs_positions_source, batch_idx):
        # get the input ids and attention masks together
        global_attention_mask = torch.zeros_like(input_ids).cuda()
        # put global attention on <s> token
        global_attention_mask[:, 0] = 1
        global_attention_mask[input_ids == self.docsep_token_id] = 1
        global_attention_mask[input_ids == self.sentsep_token_id] = 1
        attention_mask = torch.ones_like(input_ids).cuda()
        attention_mask[input_ids == self.pad_token_id] = 0
        generated_ids = self.generate_with_heterograph(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            # use_cache=False,
            max_length=self.args.max_length_tgt,
            min_length=self.args.min_length_tgt,
            num_beams=self.args.beam_size,
            no_repeat_ngram_size=3 if self.args.apply_triblck else None,
            length_penalty=self.args.length_penalty,
            heterograph=heterograph_source,
            words_positions_source=words_positions_source,
            sents_positions_source=sents_positions_source,
            docs_positions_source=docs_positions_source,

        )

        generated_str = self.tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )

        if self.args.mode == "test":
            if self.args.apply_triblck:
                output_dir = os.path.join(
                    self.args.model_path,
                    "generated_txt_%d_%s_triblck_beam=%d_%d_%d"
                    % (
                        self.args.mask_num,
                        self.args.dataset_name,
                        self.args.beam_size,
                        self.args.max_length_input,
                        self.args.max_length_tgt,
                    ),
                )
            else:
                output_dir = os.path.join(
                    self.args.model_path,
                    "generated_txt_%d_%s_beam=%d_%d_%d"
                    % (
                        self.args.mask_num,
                        self.args.dataset_name,
                        self.args.beam_size,
                        self.args.max_length_input,
                        self.args.max_length_tgt,
                    ),
                )
            if batch_idx == 0:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                else:
                    for file in os.listdir(output_dir):
                        os.remove(os.path.join(output_dir, file))
            idx = len(os.listdir(output_dir))
        result_batch = []
        if self.args.debug_mode:
            pdb.set_trace()
        for ref, pred in zip(gold_str, generated_str):
            # change <n> to \n
            pred = pred.replace("<n>", "\n")

            if self.args.mode == "test":
                with open(os.path.join(output_dir, "%d.txt" % (idx)), "w") as of:
                    of.write(pred)
                idx += 1

            s = rouge(reference=ref, candidate=pred, use_stemmer=True,
                      types=["rouge1", "rouge2", "rougeL", "rougeLsum"], split_summaries=True)
            result_batch.append(
                (
                    s["rouge1"]["recall"],
                    s["rouge1"]["precision"],
                    s["rouge1"]["fmeasure"],
                    s["rouge2"]["recall"],
                    s["rouge2"]["precision"],
                    s["rouge2"]["fmeasure"],
                    s["rougeL"]["recall"],
                    s["rougeL"]["precision"],
                    s["rougeL"]["fmeasure"],
                    s["rougeLsum"]["recall"],
                    s["rougeLsum"]["precision"],
                    s["rougeLsum"]["fmeasure"]
                )
            )
        return result_batch

    # def validation_step(self, batch, batch_idx):
    #     for p in self.model.parameters():
    #         p.requires_grad = False
    #     input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source, sents_positions_source, docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt, tgt = batch
    #     loss = self.shared_step(input_ids_source, output_ids, input_ids_summary, heterograph_source,
    #                             words_positions_source,
    #                             sents_positions_source, docs_positions_source, heterograph_tgt, words_positions_tgt,
    #                             sents_positions_tgt)
    #     # print("validation step", input_ids_source.size())
    #     if self.args.compute_rouge:
    #         result_batch = self.compute_rouge_batch(input_ids_source, tgt, heterograph_source, words_positions_source,
    #                                                 sents_positions_source, docs_positions_source, batch_idx)
    #         return {"vloss": loss, "rouge_result": result_batch}
    #     else:
    #         return {"vloss": loss}

    # def validation_epoch_end(self, outputs):
    #     for p in self.model.parameters():
    #         p.requires_grad = True

    #     vloss = torch.stack([x["vloss"] for x in outputs]).mean()
    #     self.log("vloss", vloss, sync_dist=True if self.use_ddp else False)
    #     if self.args.compute_rouge:
    #         names, metrics, avgf = self.compute_rouge_all(outputs, output_file="valid")
    #         metrics = [vloss] + metrics
    #         names = ["vloss"] + names
    #         logs = dict(zip(*[names, metrics]))
    #         self.logger.log_metrics(logs, step=self.global_step)
    #         self.log("avgf", avgf)
    #         return {
    #             "avg_val_loss": vloss,
    #             "avgf": avgf,
    #             "log": logs,
    #             "progress_bar": logs,
    #         }
    #     else:
    #         logs = {"vloss": vloss}
    #         self.logger.log_metrics(logs, step=self.global_step)
    #         return {"vloss": vloss, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        for p in self.model.parameters():
            p.requires_grad = False
        input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source, sents_positions_source, docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt, tgt = batch
        loss = self.shared_step(input_ids_source, output_ids, input_ids_summary, heterograph_source,
                                words_positions_source,
                                sents_positions_source, docs_positions_source, heterograph_tgt, words_positions_tgt,
                                sents_positions_tgt)
        
        if self.args.compute_rouge:
            result_batch = self.compute_rouge_batch(input_ids_source, tgt, heterograph_source, words_positions_source,
                                                    sents_positions_source, docs_positions_source, batch_idx)
            output = {"vloss": loss, "rouge_result": result_batch}
        else:
            output = {"vloss": loss}
        
        # Lưu output vào instance attribute
        self.validation_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        for p in self.model.parameters():
            p.requires_grad = True

        if not self.validation_outputs:
            return

        vloss = torch.stack([x["vloss"] for x in self.validation_outputs]).mean()
        self.log("vloss", vloss, sync_dist=True if self.use_ddp else False)
        
        if self.args.compute_rouge:
            names, metrics, avgf = self.compute_rouge_all(self.validation_outputs, output_file="valid")
            metrics = [vloss] + metrics
            names = ["vloss"] + names
            logs = dict(zip(*[names, metrics]))
            self.logger.log_metrics(logs, step=self.global_step)
            self.log("avgf", avgf)
            return {
                "avg_val_loss": vloss,
                "avgf": avgf,
                "log": logs,
                "progress_bar": logs,
            }
        else:
            logs = {"vloss": vloss}
            self.logger.log_metrics(logs, step=self.global_step)
            return {"vloss": vloss, "log": logs, "progress_bar": logs}
        
        # Xóa outputs sau mỗi epoch
        self.validation_outputs.clear()

    def compute_rouge_all(self, outputs, output_file=None):
        rouge_result_all = [r for b in outputs for r in b["rouge_result"]]
        names = []
        for rouge in ["1", "2", "L", "Lsum"]:
            names.extend(
                [
                    "rouge-{}-r".format(rouge),
                    "rouge-{}-p".format(rouge),
                    "rouge-{}-f".format(rouge),
                ]
            )
        rouge_results = pd.DataFrame(rouge_result_all, columns=names)
        avg = [rouge_results[c].mean() for c in rouge_results.columns]
        rouge_results.loc["avg_score"] = avg
        if output_file:
            csv_name = (
                    self.args.model_path
                    + output_file
                    + "-%d.csv" % (torch.distributed.get_rank() if self.use_ddp else 0)
            )
            rouge_results.to_csv(csv_name)

        avgf = (avg[2] + avg[5] + avg[11]) / 3
        metrics = avg
        print("Validation Result at Step %d" % (self.global_step))
        print(
            "Rouge-1 r score: %f, Rouge-1 p score: %f, Rouge-1 f-score: %f"
            % (metrics[0], metrics[1], metrics[2])
        )
        print(
            "Rouge-2 r score: %f, Rouge-2 p score: %f, Rouge-2 f-score: %f"
            % (metrics[3], metrics[4], metrics[5])
        )
        print(
            "Rouge-L r score: %f, Rouge-L p score: %f, Rouge-L f-score: %f"
            % (metrics[6], metrics[7], metrics[8])
        )
        print(
            "Rouge-Lsum r score: %f, Rouge-Lsum p score: %f, Rouge-Lsum f-score: %f"
            % (metrics[9], metrics[10], metrics[11])
        )
        return names, metrics, avgf

    def test_step(self, batch, batch_idx):
        output = self.validation_step(batch, batch_idx)
        self.test_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return
            
        tloss = torch.stack([x["vloss"] for x in self.test_outputs]).mean()
        self.log("tloss", tloss, sync_dist=True if self.use_ddp else False)
        
        output_file = "test_%s_%d_%d_beam=%d_lenPen=%.2f" % (
            self.args.dataset_name,
            self.args.max_length_input,
            self.args.max_length_tgt,
            self.args.beam_size,
            self.args.length_penalty,
        )
        output_file = (
            output_file
            + "_fewshot_%d_%d" % (self.args.num_train_data, self.args.rand_seed)
            if self.args.fewshot
            else output_file
        )
        
        names, metrics, avgf = self.compute_rouge_all(self.test_outputs, output_file=output_file)
        metrics = [tloss, avgf] + metrics
        names = ["tloss", "avgf"] + names
        logs = dict(zip(*[names, metrics]))
        self.logger.log_metrics(logs, step=self.global_step)
        self.log("avgf", avgf)
        
        # Xóa outputs sau khi test
        self.test_outputs.clear()
        return {"avg_test_loss": tloss, "avgf": avgf, "log": logs, "progress_bar": logs}

    def generate_with_heterograph(self, input_ids, attention_mask=None, global_attention_mask=None,
                             heterograph=None, words_positions_source=None, 
                             sents_positions_source=None, docs_positions_source=None,
                             max_length=142, min_length=56, num_beams=4, 
                             no_repeat_ngram_size=3, length_penalty=2.0, 
                             early_stopping=True, do_sample=False):
        """
        Custom generate method for LED model with heterograph context
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Prepare decoder input ids
        decoder_start_token_id = getattr(self.model.config, 'decoder_start_token_id', self.model.config.bos_token_id)
        if decoder_start_token_id is None:
            decoder_start_token_id = self.model.config.eos_token_id
        
        decoder_input_ids = torch.full(
            (batch_size, 1), 
            decoder_start_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        # Prepare base model inputs cho encoder
        base_model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        if global_attention_mask is not None:
            base_model_inputs['global_attention_mask'] = global_attention_mask
        
        # Thêm heterograph parameters nếu có
        if heterograph is not None:
            base_model_inputs['heterograph'] = heterograph
        if words_positions_source is not None:
            base_model_inputs['words_positions_source'] = words_positions_source
        if sents_positions_source is not None:
            base_model_inputs['sents_positions_source'] = sents_positions_source
        if docs_positions_source is not None:
            base_model_inputs['docs_positions_source'] = docs_positions_source
        
        # Generation
        if num_beams > 1:
            return self._generate_beam_search(
                base_model_inputs=base_model_inputs,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
                early_stopping=early_stopping
            )
        else:
            return self._generate_greedy(
                base_model_inputs=base_model_inputs,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample
            )

    def _generate_beam_search(self, base_model_inputs, decoder_input_ids, max_length, min_length, 
                            num_beams, no_repeat_ngram_size, length_penalty, early_stopping):
        """
        Beam search generation - gọi model đầy đủ mỗi step
        """
        device = decoder_input_ids.device
        batch_size = decoder_input_ids.shape[0]
        
        # Expand inputs for beam search
        expanded_batch_idxs = torch.arange(batch_size, device=device).view(-1, 1).repeat(1, num_beams).view(-1)
        
        # Expand tất cả inputs
        beam_model_inputs = {}
        for key, value in base_model_inputs.items():
            if isinstance(value, torch.Tensor):
                beam_model_inputs[key] = value.index_select(0, expanded_batch_idxs)
            else:
                # Cho các tham số như heterograph (có thể là object)
                beam_model_inputs[key] = value
        
        # Initialize beams
        beam_size = batch_size * num_beams
        decoder_input_ids = decoder_input_ids.repeat(num_beams, 1)
        
        # Beam scores
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)
        
        # Keep track of finished sequences
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generation loop
        cur_len = decoder_input_ids.shape[1]
        while cur_len < max_length:
            # Prepare full model inputs
            model_inputs = beam_model_inputs.copy()
            model_inputs['decoder_input_ids'] = decoder_input_ids
            model_inputs['use_cache'] = False
            model_inputs['return_dict'] = True
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**model_inputs)
            
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply length penalty
            if length_penalty != 1.0:
                next_token_logits = next_token_logits / (cur_len ** length_penalty)
            
            # Apply no repeat ngram
            if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
                next_token_logits = self._apply_no_repeat_ngram(
                    decoder_input_ids, next_token_logits, no_repeat_ngram_size
                )
            
            # Get top k candidates
            vocab_size = next_token_logits.shape[-1]
            next_token_scores = torch.log_softmax(next_token_logits, dim=-1)
            next_token_scores = next_token_scores + beam_scores[:, None]
            
            # Reshape for beam search
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            
            # Get top 2*num_beams candidates
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            
            # Process each batch
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode='floor')
            next_tokens = next_tokens % vocab_size
            
            # Update beams
            beam_outputs = []
            beam_scores_new = []
            
            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # Keep existing beams for finished sequences
                    beam_outputs.extend([
                        decoder_input_ids[batch_idx * num_beams + i] 
                        for i in range(num_beams)
                    ])
                    beam_scores_new.extend([
                        beam_scores[batch_idx * num_beams + i] 
                        for i in range(num_beams)
                    ])
                    continue
                
                batch_beam_outputs = []
                batch_beam_scores = []
                
                for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                    zip(next_tokens[batch_idx], next_token_scores[batch_idx], next_indices[batch_idx])
                ):
                    beam_id = batch_idx * num_beams + next_index
                    
                    # Check if we have enough beams
                    if len(batch_beam_outputs) >= num_beams:
                        break
                    
                    # Add token to beam
                    new_beam = torch.cat([decoder_input_ids[beam_id], next_token.unsqueeze(0)])
                    batch_beam_outputs.append(new_beam)
                    batch_beam_scores.append(next_score)
                
                beam_outputs.extend(batch_beam_outputs)
                beam_scores_new.extend(batch_beam_scores)
            
            # Update decoder_input_ids and beam_scores
            max_len = max(len(beam) for beam in beam_outputs)
            decoder_input_ids = torch.full(
                (len(beam_outputs), max_len), 
                self.model.config.pad_token_id, 
                dtype=torch.long, 
                device=device
            )
            
            for i, beam in enumerate(beam_outputs):
                decoder_input_ids[i, :len(beam)] = beam
            
            beam_scores = torch.tensor(beam_scores_new, device=device)
            cur_len = decoder_input_ids.shape[1]
            
            # Check for early stopping
            if early_stopping and cur_len >= min_length:
                # Check if any beam ends with EOS
                eos_token_id = self.model.config.eos_token_id
                if eos_token_id is not None:
                    for batch_idx in range(batch_size):
                        if not done[batch_idx]:
                            batch_beams = decoder_input_ids[batch_idx * num_beams:(batch_idx + 1) * num_beams]
                            if any(beam[-1] == eos_token_id for beam in batch_beams):
                                done[batch_idx] = True
                    
                    if done.all():
                        break
        
        # Return best beams for each batch
        final_outputs = []
        for batch_idx in range(batch_size):
            batch_beams = decoder_input_ids[batch_idx * num_beams:(batch_idx + 1) * num_beams]
            batch_scores = beam_scores[batch_idx * num_beams:(batch_idx + 1) * num_beams]
            best_beam_idx = torch.argmax(batch_scores)
            final_outputs.append(batch_beams[best_beam_idx])
        
        # Stack and pad to same length
        max_len = max(len(output) for output in final_outputs)
        result = torch.full(
            (batch_size, max_len), 
            self.model.config.pad_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        for i, output in enumerate(final_outputs):
            result[i, :len(output)] = output
        
        return result

    def _generate_greedy(self, base_model_inputs, decoder_input_ids, max_length, min_length, do_sample):
        """
        Greedy generation - gọi model đầy đủ mỗi step
        """
        device = decoder_input_ids.device
        batch_size = decoder_input_ids.shape[0]
        
        cur_len = decoder_input_ids.shape[1]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        while cur_len < max_length:
            # Prepare full model inputs
            model_inputs = base_model_inputs.copy()
            model_inputs['decoder_input_ids'] = decoder_input_ids
            model_inputs['use_cache'] = False
            model_inputs['return_dict'] = True
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**model_inputs)
            
            next_token_logits = outputs.logits[:, -1, :]
            
            if do_sample:
                # Sampling
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Update sequences
            next_tokens = next_tokens * unfinished_sequences + self.model.config.pad_token_id * (1 - unfinished_sequences)
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens[:, None]], dim=1)
            
            # Check for EOS
            if self.model.config.eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.ne(self.model.config.eos_token_id).long()
                )
            
            cur_len += 1
            
            # Early stopping
            if unfinished_sequences.max() == 0:
                break
        
        return decoder_input_ids

    def _apply_no_repeat_ngram(self, input_ids, logits, no_repeat_ngram_size):
        """
        Apply no repeat ngram constraint
        """
        batch_size, cur_len = input_ids.shape
        
        if cur_len + 1 < no_repeat_ngram_size:
            return logits
        
        for batch_idx in range(batch_size):
            banned_tokens = set()
            for i in range(cur_len - no_repeat_ngram_size + 2):
                ngram = tuple(input_ids[batch_idx, i:i + no_repeat_ngram_size - 1].tolist())
                banned_tokens.add(input_ids[batch_idx, i + no_repeat_ngram_size - 1].item())
            
            for token in banned_tokens:
                logits[batch_idx, token] = -float('inf')
        
        return logits


def train(args):
    model = HGSummarizer(args)

    # load dataset
    train_dataloader = get_dataloader_summ(args, model.tokenizer, 'train', args.num_workers, True)
    valid_dataloader = get_dataloader_summ(args, model.tokenizer, 'validation', args.num_workers, False)

    # initialize checkpoint
    if args.ckpt_path is None:
        args.ckpt_path = os.path.join(args.model_path, "summ_checkpoints/")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        filename="{step}-{vloss:.2f}-{avgf:.4f}",
        save_top_k=args.save_top_k,
        monitor="avgf",
        mode="max",
        save_on_train_epoch_end=False,
    )
    early_stopping = EarlyStopping(monitor='vloss', patience=3, mode='min')

    # initialize logger
    logger = TensorBoardLogger(args.model_path + "tb_logs", name=args.model_name)

    # initialize trainer
    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator=args.accelerator,
        # auto_select_gpus=True,
        strategy=args.speed_strategy,
        # track_grad_norm=-1,
        max_steps=args.total_steps * args.accum_batch,
        # replace_sampler_ddp=False,
        accumulate_grad_batches=args.accum_batch,
        # val_check_interval=0.5,
        check_val_every_n_epoch=1 if args.num_train_data > 100 else 5,
        logger=logger,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, early_stopping],
        # lỗi cuda nên phải giảm từ 32 xuống 16
        precision=16,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches else 1.0,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches else 1.0,
        num_sanity_val_steps=0
    )

    # pdb.set_trace()
    trainer.fit(model, train_dataloader, valid_dataloader)

    if args.test_imediate:
        args.resume_ckpt = checkpoint_callback.best_model_path
        print(args.resume_ckpt)
        if args.test_batch_size != -1:
            args.batch_size = args.test_batch_size
        args.mode = "test"
        test(args)


def test(args):
    # initialize trainer
    trainer = pl.Trainer(
        devices=1,
        # auto_select_gpus=True,
        accelerator=args.accelerator,
        # track_grad_norm=-1,
        max_steps=args.total_steps * args.accum_batch,
        # replace_sampler_ddp=False,
        log_every_n_steps=5,
        # lỗi cuda nên phải giảm từ 32 xuống 16
        precision=16,
        limit_test_batches=args.limit_test_batches if args.limit_test_batches else 1.0
    )

    if args.resume_ckpt is not None:
        model = HGSummarizer.load_from_checkpoint(args.resume_ckpt, args=args)
    else:
        model = HGSummarizer(args)

    # load dataset
    test_dataloader = get_dataloader_summ(args, model.tokenizer, 'test', args.num_workers, False)

    # test
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    seed_everything(42, workers=True)
    parser = argparse.ArgumentParser()

    # Gneral
    parser.add_argument("--gpus", default=1, type=int, help="The number of gpus to use")
    parser.add_argument("--accelerator", default="gpu", type=str, choices=["gpu", "cpu"])
    parser.add_argument("--speed_strategy", default=None, type=str, help="Accelerator strategy, e.g., ddp")
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument("--model_name", default="HGSum")
    parser.add_argument("--pretrained_primer", type=str, default=None,
                        help="Name or path of pretrained PRIMERA from Huggingface, or the model to be tested")
    parser.add_argument("--with_sent_sep", action="store_true",
                        help="Insert <sent-sep> at the end of each sentence when concatenating different documents")
    parser.add_argument("--debug_mode", action="store_true", help="Set true if to debug")
    parser.add_argument("--compute_rouge", action="store_true", help="whether to compute rouge in validation steps")
    parser.add_argument("--progress_bar_refresh_rate", default=1, type=int)
    parser.add_argument("--model_path", type=str, default="result/",
                        help="The path to save output and checkpoints in training and testing")
    parser.add_argument("--ckpt_path", type=str, default=None, help="dir to save checkpoints")
    parser.add_argument("--save_top_k", default=3, type=int)
    parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from", default=None)
    parser.add_argument("--data_path", type=str, default="../../datasets/")
    parser.add_argument("--dataset_name", type=str, default="multinews",
                        choices=["multinews", "arxiv", "multixscience", "wcep_10", "wcep_100", "peersum_r",
                                 "peersum_rc", "peersum_all"])
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use for dataloader")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_length_input", default=4096, type=int)
    parser.add_argument("--max_length_tgt", default=1024, type=int)
    parser.add_argument("--min_length_tgt", default=0, type=int)
    parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
    parser.add_argument("--adafactor", action="store_true", help="Use adafactor optimizer")
    parser.add_argument("--grad_ckpt", action="store_true", help="Enable gradient checkpointing to save memory")
    parser.add_argument("--rand_seed", type=int, default=42,
                        help="Seed for random sampling, useful for few shot learning")

    # For training
    parser.add_argument("--limit_train_batches", type=float, default=1.0, help="Use limited batches in training")
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help="Use limited batches in validation")
    parser.add_argument("--lr", type=float, default=3e-5, help="Maximum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--accum_data_per_step", type=int, default=16, help="Number of data per step")
    parser.add_argument("--total_steps", type=int, default=50000, help="Number of steps to train")
    parser.add_argument("--num_train_data", type=int, default=-1,
                        help="Number of training data, -1 for full dataset and any positive number indicates how many data to use")
    parser.add_argument("--fix_lr", action="store_true", help="use fix learning rate")
    parser.add_argument("--test_imediate", action="store_true", help="test on the best checkpoint")
    parser.add_argument("--fewshot", action="store_true", help="whether this is a run for few shot learning")

    # For testing
    parser.add_argument("--limit_test_batches", type=float, default=1.0,
                        help="Number of batches to test in the test mode")
    parser.add_argument("--beam_size", type=int, default=1, help="size of beam search")
    parser.add_argument("--length_penalty", type=float, default=1, help="length penalty of generated text")
    parser.add_argument("--mask_num", type=int, default=0, help="Number of masks in the input of summarization data")
    parser.add_argument("--test_batch_size", type=int, default=-1,
                        help="batch size for test, used in few shot evaluation.")
    parser.add_argument("--apply_triblck", action="store_true",
                        help="whether apply trigram block in the evaluation phase")
    parser.add_argument("--num_test_data", type=int, default=-1, help="the number of testing data")

    args = parser.parse_args()
    args.accum_batch = args.accum_data_per_step // args.batch_size

    if args.gpus > 0:
        args.accelerator = "gpu"
    else:
        args.accelerator = "cpu"
        args.gpus = 1
    if args.num_workers == -1:
        args.num_workers = os.cpu_count()
    print(args)

    if not os.path.exists(args.model_path):  # this is used to save the checkpoints and logs
        os.makedirs(args.model_path)
    with open(os.path.join(args.model_path, "args_%s_%s.json" % (args.mode, args.dataset_name)), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    if args.mode == "train":
        train(args)
    if args.mode == "test":
        test(args)
