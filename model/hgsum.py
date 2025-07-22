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

import torch
import torch.nn.functional as F

def safe_index_select(tensor, positions, dim=0, tensor_name="tensor"):
    """
    Safely index select tensor using positions, ensuring no out-of-bounds access
    """
    # Convert to tensor if it's a list
    if isinstance(positions, list):
        positions = torch.tensor(positions, device=tensor.device, dtype=torch.long)
    elif not isinstance(positions, torch.Tensor):
        positions = torch.tensor(positions, device=tensor.device, dtype=torch.long)
    
    if len(positions) == 0:
        print(f"WARNING: positions is empty for {tensor_name}")
        if dim == 0:
            return torch.empty(0, tensor.shape[1] if tensor.ndim > 1 else tensor.shape[0], 
                              dtype=tensor.dtype, device=tensor.device)
        else:
            return torch.empty(tensor.shape[0], 0, 
                              dtype=tensor.dtype, device=tensor.device)
    
    # Ensure positions is on the same device as tensor
    positions = positions.to(tensor.device, dtype=torch.long)
    
    # Validate positions
    max_idx = tensor.shape[dim] - 1
    min_idx = 0
    
    # Check for out-of-bounds indices
    invalid_mask = (positions > max_idx) | (positions < min_idx)
    
    if invalid_mask.any():
        print(f"ERROR: {tensor_name} index out of bounds detected!")
        print(f"Tensor shape[{dim}]: {tensor.shape[dim]}")
        print(f"Valid range: [0, {max_idx}]")
        print(f"Invalid positions: {positions[invalid_mask]}")
        print(f"Positions range: [{positions.min().item()}, {positions.max().item()}]")
        
        # Clamp positions to valid range
        positions = torch.clamp(positions, min_idx, max_idx)
        print(f"Positions clamped to valid range")
    
    # Remove duplicates and sort for efficiency
    positions = torch.unique(positions)
    
    try:
        return tensor.index_select(dim, positions)
    except RuntimeError as e:
        print(f"CUDA error during index_select for {tensor_name}:")
        print(f"Tensor shape: {tensor.shape}")
        print(f"Positions shape: {positions.shape}")
        print(f"Positions: {positions}")
        raise e

def validate_and_fix_positions(seq_length, positions, tensor_name="tensor", positions_name="positions"):
    """
    Validate that positions are within sequence bounds and fix if necessary
    """
    # CRITICAL FIX: Convert list to tensor first
    if isinstance(positions, list):
        if len(positions) == 0:
            print(f"WARNING: {positions_name} is empty list for {tensor_name}")
            return torch.tensor([], dtype=torch.long)
        positions = torch.tensor(positions, dtype=torch.long)
    elif not isinstance(positions, torch.Tensor):
        positions = torch.tensor(positions, dtype=torch.long)
    
    # Ensure correct dtype
    positions = positions.long()
    
    if len(positions) == 0:
        print(f"WARNING: {positions_name} is empty for {tensor_name}")
        return positions
    
    max_valid_idx = seq_length - 1  # Use actual sequence length
    
    print(f"DEBUG: Validating {positions_name} for {tensor_name}")
    print(f"  - Sequence length: {seq_length}")
    print(f"  - Max valid index: {max_valid_idx}")
    print(f"  - Positions range: [{positions.min().item()}, {positions.max().item()}]")
    print(f"  - Total positions: {len(positions)}")
    
    # Check for negative indices
    negative_mask = positions < 0
    if negative_mask.any():
        print(f"ERROR: {positions_name} contains negative indices: {positions[negative_mask]}")
        positions[negative_mask] = 0
    
    # Check for out-of-bounds indices
    invalid_mask = positions > max_valid_idx
    
    if invalid_mask.any():
        print(f"ERROR: {positions_name} contains out-of-bounds indices!")
        print(f"Invalid positions: {positions[invalid_mask]}")
        print(f"Invalid count: {invalid_mask.sum().item()}")
        
        # Clamp invalid positions to max valid index
        positions[invalid_mask] = max_valid_idx
        print(f"Invalid positions clamped to {max_valid_idx}")
    
    # Final validation
    assert positions.min().item() >= 0, f"Still have negative indices in {positions_name}"
    assert positions.max().item() <= max_valid_idx, f"Still have out-of-bounds indices in {positions_name}"
    
    return positions

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

    # def forward(self, input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source,
    #             sents_positions_source,
    #             docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt):
    #     device = input_ids_source.device
    #     # print("encoding source", len(heterograph_source))
    #     decoder_input_ids = output_ids[:, :-1]
    #     # get the input ids and attention masks together
    #     global_attention_mask_source = torch.zeros_like(input_ids_source).to(device)
    #     # put global attention on <s> token
    #     global_attention_mask_source[:, 0] = 1
    #     # put global attention on <doc-sep> token
    #     global_attention_mask_source[input_ids_source == self.docsep_token_id] = 1
    #     # put global attention on <sent-sep> token
    #     global_attention_mask_source[input_ids_source == self.sentsep_token_id] = 1

    #     attention_mask = torch.ones_like(input_ids_source).to(device)
    #     attention_mask = attention_mask.type_as(input_ids_source)
    #     attention_mask[input_ids_source == self.pad_token_id] = 0
    #     # encoding source documents
    #     outputs_source = self.model(
    #         input_ids_source,
    #         attention_mask=attention_mask,
    #         decoder_input_ids=decoder_input_ids,
    #         global_attention_mask=global_attention_mask_source,
    #         use_cache=False,
    #         heterograph=heterograph_source,
    #         words_positions_source=words_positions_source,
    #         sents_positions_source=sents_positions_source,
    #         docs_positions_source=docs_positions_source
    #     )
    #     lm_logits = outputs_source.logits
    #     assert lm_logits.shape[-1] == self.model.config.vocab_size

    #     # print("encoding summary", input_ids_summary.size())
    #     # pdb.set_trace()
    #     # encoding summary
    #     # get the input ids and attention masks together
    #     global_attention_mask_summary = torch.zeros_like(input_ids_summary).to(device)
    #     # put global attention on <s> token
    #     global_attention_mask_summary[:, 0] = 1
    #     # put global attention on <sent-sep> token
    #     global_attention_mask_summary[input_ids_summary == self.sentsep_token_id] = 1
    #     outputs_summary = self.model(
    #         input_ids_summary.to(device),
    #         decoder_input_ids=None,
    #         global_attention_mask=global_attention_mask_summary,
    #         use_cache=False,
    #         heterograph=heterograph_tgt,
    #         words_positions_source=words_positions_tgt,
    #         sents_positions_source=sents_positions_tgt,
    #         docs_positions_source=None
    #     )

    #     return lm_logits, outputs_source.mgat_outputs, outputs_source.sagpooling_outputs, outputs_summary.mgat_outputs
        # return lm_logits, outputs_source.mgat_outputs, outputs_source.sagpooling_outputs, None

    def forward(self, input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source,
            sents_positions_source, docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt):
        device = input_ids_source.device
        
        # Debug input information
        print(f"DEBUG: input_ids_source shape: {input_ids_source.shape}")
        print(f"DEBUG: input_ids_summary shape: {input_ids_summary.shape}")
        
        # Get actual sequence lengths (excluding padding)
        source_seq_length = input_ids_source.shape[1]
        summary_seq_length = input_ids_summary.shape[1]
        
        print(f"DEBUG: Source sequence length: {source_seq_length}")
        print(f"DEBUG: Summary sequence length: {summary_seq_length}")
        
        # Handle positions - convert lists to tensors and validate strictly
        def process_positions(positions, name, seq_length):
            if positions is None:
                return None
            
            if isinstance(positions, list):
                # Handle list of tensors (batch)
                if len(positions) > 0 and isinstance(positions[0], torch.Tensor):
                    # This is a batch - take the first item for now
                    positions = positions[0]
                elif isinstance(positions, list):
                    # This is a simple list
                    if len(positions) == 0:
                        return torch.tensor([], dtype=torch.long, device=device)
                    positions = torch.tensor(positions, dtype=torch.long, device=device)
            
            if isinstance(positions, torch.Tensor):
                positions = positions.to(device, dtype=torch.long)
            
            # CRITICAL: Validate against actual sequence length
            positions = validate_and_fix_positions(seq_length, positions, f"sequence_{name}", name)
            
            print(f"DEBUG: {name} final - len: {len(positions)}, range: [{positions.min().item() if len(positions) > 0 else 'N/A'}, {positions.max().item() if len(positions) > 0 else 'N/A'}]")
            return positions
        
        # Process all positions with their respective sequence lengths
        words_positions_source = process_positions(words_positions_source, "words_positions_source", source_seq_length)
        sents_positions_source = process_positions(sents_positions_source, "sents_positions_source", source_seq_length)
        docs_positions_source = process_positions(docs_positions_source, "docs_positions_source", source_seq_length)
        words_positions_tgt = process_positions(words_positions_tgt, "words_positions_tgt", summary_seq_length)
        sents_positions_tgt = process_positions(sents_positions_tgt, "sents_positions_tgt", summary_seq_length)
        
        decoder_input_ids = output_ids[:, :-1]
        
        # Get the input ids and attention masks together
        global_attention_mask_source = torch.zeros_like(input_ids_source).to(device)
        # Put global attention on <s> token
        global_attention_mask_source[:, 0] = 1
        # Put global attention on <doc-sep> token
        global_attention_mask_source[input_ids_source == self.docsep_token_id] = 1
        # Put global attention on <sent-sep> token
        global_attention_mask_source[input_ids_source == self.sentsep_token_id] = 1

        attention_mask = torch.ones_like(input_ids_source).to(device)
        attention_mask = attention_mask.type_as(input_ids_source)
        attention_mask[input_ids_source == self.pad_token_id] = 0
        
        # Additional safety check before model forward
        if words_positions_source is not None and len(words_positions_source) > 0:
            max_pos = words_positions_source.max().item()
            if max_pos >= source_seq_length:
                print(f"CRITICAL ERROR: words_positions_source max ({max_pos}) >= seq_length ({source_seq_length})")
                words_positions_source = torch.clamp(words_positions_source, 0, source_seq_length - 1)
        
        if sents_positions_source is not None and len(sents_positions_source) > 0:
            max_pos = sents_positions_source.max().item()
            if max_pos >= source_seq_length:
                print(f"CRITICAL ERROR: sents_positions_source max ({max_pos}) >= seq_length ({source_seq_length})")
                sents_positions_source = torch.clamp(sents_positions_source, 0, source_seq_length - 1)
        
        if docs_positions_source is not None and len(docs_positions_source) > 0:
            max_pos = docs_positions_source.max().item()
            if max_pos >= source_seq_length:
                print(f"CRITICAL ERROR: docs_positions_source max ({max_pos}) >= seq_length ({source_seq_length})")
                docs_positions_source = torch.clamp(docs_positions_source, 0, source_seq_length - 1)
        
        print("DEBUG: Final validation passed, calling model...")
        
        try:
            # Encoding source documents
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
        except RuntimeError as e:
            print(f"ERROR in source encoding:")
            print(f"input_ids_source shape: {input_ids_source.shape}")
            print(f"attention_mask shape: {attention_mask.shape}")
            print(f"decoder_input_ids shape: {decoder_input_ids.shape}")
            print(f"words_positions_source: {words_positions_source}")
            print(f"sents_positions_source: {sents_positions_source}")
            print(f"docs_positions_source: {docs_positions_source}")
            
            # Additional debug for embeddings
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'embed_tokens'):
                vocab_size = self.model.encoder.embed_tokens.num_embeddings
                print(f"Model vocab size: {vocab_size}")
                print(f"input_ids_source range: [{input_ids_source.min().item()}, {input_ids_source.max().item()}]")
            
            raise e
        
        lm_logits = outputs_source.logits
        assert lm_logits.shape[-1] == self.model.config.vocab_size

        # Encoding summary - similarly validate for target positions
        if words_positions_tgt is not None and len(words_positions_tgt) > 0:
            max_pos = words_positions_tgt.max().item()
            if max_pos >= summary_seq_length:
                print(f"CRITICAL ERROR: words_positions_tgt max ({max_pos}) >= seq_length ({summary_seq_length})")
                words_positions_tgt = torch.clamp(words_positions_tgt, 0, summary_seq_length - 1)
        
        if sents_positions_tgt is not None and len(sents_positions_tgt) > 0:
            max_pos = sents_positions_tgt.max().item()
            if max_pos >= summary_seq_length:
                print(f"CRITICAL ERROR: sents_positions_tgt max ({max_pos}) >= seq_length ({summary_seq_length})")
                sents_positions_tgt = torch.clamp(sents_positions_tgt, 0, summary_seq_length - 1)

        # Get the input ids and attention masks together
        global_attention_mask_summary = torch.zeros_like(input_ids_summary).to(device)
        # Put global attention on <s> token
        global_attention_mask_summary[:, 0] = 1
        # Put global attention on <sent-sep> token
        global_attention_mask_summary[input_ids_summary == self.sentsep_token_id] = 1
        
        try:
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
        except RuntimeError as e:
            print(f"ERROR in summary encoding:")
            print(f"input_ids_summary shape: {input_ids_summary.shape}")
            print(f"words_positions_tgt: {words_positions_tgt}")
            print(f"sents_positions_tgt: {sents_positions_tgt}")
            raise e

        return lm_logits, outputs_source.mgat_outputs, outputs_source.sagpooling_outputs, outputs_summary.mgat_outputs


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

    # def shared_step(self, input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source,
    #                 sents_positions_source,
    #                 docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt):
        
    #     # Debug shapes
    #     print(f"input_ids_source shape: {input_ids_source.shape}")
    #     print(f"output_ids shape: {output_ids.shape}")
    #     print(f"output_ids min/max: {output_ids.min()}/{output_ids.max()}")
        
    #     # Kiểm tra có giá trị âm không
    #     if (output_ids < 0).any():
    #         print("Warning: Negative values in output_ids")

    #     # Debug kiểu dữ liệu
    #     print(f"Type of heterograph_source: {type(heterograph_source)}")
    #     print(f"Content of heterograph_source: {heterograph_source}")
        
    #     # Nếu là tuple, kiểm tra nội dung
    #     if isinstance(heterograph_source, tuple):
    #         print(f"Tuple length: {len(heterograph_source)}")
    #         for i, item in enumerate(heterograph_source):
    #             print(f"Item {i}: {type(item)}, shape: {getattr(item, 'shape', 'No shape')}")

    #     lm_logits, mgat_outputs_source, sagpooling_ouputs, mgat_outputs_summary = self.forward(input_ids_source,
    #                                                                                            output_ids,
    #                                                                                            input_ids_summary,
    #                                                                                            heterograph_source,
    #                                                                                            words_positions_source,
    #                                                                                            sents_positions_source,
    #                                                                                            docs_positions_source,
    #                                                                                            heterograph_tgt,
    #                                                                                            words_positions_tgt,
    #                                                                                            sents_positions_tgt)

    #     # print("shared step")
    #     # graph similarity loss
    #     cos = torch.nn.CosineSimilarity(dim=1)
    #     graph_sim = torch.mean(cos(sagpooling_ouputs, mgat_outputs_summary))
    #     # coss-entropy loss
    #     labels = output_ids[:, 1:].clone()

    #     if self.args.label_smoothing == 0.0:
    #         # Same behavior as modeling_bart.py, besides ignoring pad_token_id
    #         ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
    #         loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
    #     else:
    #         lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
    #         loss, nll_loss = label_smoothed_nll_loss(
    #             lprobs,
    #             labels,
    #             self.args.label_smoothing,
    #             ignore_index=self.pad_token_id,
    #         )
    #     if torch.isnan(loss):
    #         pdb.set_trace()

    #     return 0.5 * loss + 0.5 * graph_sim

    def shared_step(self, input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source,
                sents_positions_source, docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt):
    
        # Debug shapes
        print(f"input_ids_source shape: {input_ids_source.shape}")
        print(f"output_ids shape: {output_ids.shape}")
        print(f"input_ids_summary shape: {input_ids_summary.shape}")
        
        # CRITICAL: Validate all input tensors for vocab bounds
        def validate_input_ids(tensor, name):
            if tensor is None:
                return tensor
                
            # Check for negative values
            if (tensor < 0).any():
                print(f"WARNING: Negative values in {name} detected, clamping to 0")
                tensor = torch.clamp(tensor, min=0)
            
            # Check vocab bounds
            if hasattr(self.model.config, 'vocab_size'):
                vocab_size = self.model.config.vocab_size
                if (tensor >= vocab_size).any():
                    print(f"WARNING: {name} contains indices >= vocab_size ({vocab_size})")
                    invalid_count = (tensor >= vocab_size).sum().item()
                    print(f"Invalid tokens count: {invalid_count}")
                    tensor = torch.clamp(tensor, max=vocab_size-1)
            
            print(f"{name} validated - min/max: {tensor.min()}/{tensor.max()}")
            return tensor
        
        # Validate all input_ids tensors
        input_ids_source = validate_input_ids(input_ids_source, "input_ids_source")
        output_ids = validate_input_ids(output_ids, "output_ids")
        input_ids_summary = validate_input_ids(input_ids_summary, "input_ids_summary")
        
        # Debug heterograph
        print(f"Type of heterograph_source: {type(heterograph_source)}")
        print(f"Type of heterograph_tgt: {type(heterograph_tgt)}")
        
        try:
            lm_logits, mgat_outputs_source, sagpooling_ouputs, mgat_outputs_summary = self.forward(
                input_ids_source, output_ids, input_ids_summary, heterograph_source,
                words_positions_source, sents_positions_source, docs_positions_source,
                heterograph_tgt, words_positions_tgt, sents_positions_tgt
            )
        except RuntimeError as e:
            print(f"CUDA error in forward pass:")
            print(f"Error: {e}")
            
            # Enhanced debug information
            print("=== DEBUG INFO ===")
            print(f"Device: {input_ids_source.device}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA version: {torch.version.cuda}")
            
            if words_positions_source is not None:
                print(f"words_positions_source - device: {words_positions_source.device}, dtype: {words_positions_source.dtype}")
                print(f"words_positions_source - shape: {words_positions_source.shape}")
            
            # Enable CUDA debugging (add this to your training script)
            import os
            os.environ['TORCH_USE_CUDA_DSA'] = '1'
            print("TORCH_USE_CUDA_DSA enabled for detailed error info")
            
            raise e

        # Graph similarity loss
        cos = torch.nn.CosineSimilarity(dim=1)
        
        # Validate tensors before computing similarity
        if sagpooling_ouputs is None or mgat_outputs_summary is None:
            print("WARNING: One of the outputs is None, setting graph_sim to 0")
            graph_sim = torch.tensor(0.0, device=input_ids_source.device)
        else:
            # Check tensor shapes before computing similarity
            if sagpooling_ouputs.shape != mgat_outputs_summary.shape:
                print(f"WARNING: Shape mismatch - sagpooling: {sagpooling_ouputs.shape}, mgat_summary: {mgat_outputs_summary.shape}")
                # Pad or truncate to match
                min_size = min(sagpooling_ouputs.shape[0], mgat_outputs_summary.shape[0])
                sagpooling_ouputs = sagpooling_ouputs[:min_size]
                mgat_outputs_summary = mgat_outputs_summary[:min_size]
            
            # Additional validation for similarity computation
            if torch.isnan(sagpooling_ouputs).any() or torch.isnan(mgat_outputs_summary).any():
                print("WARNING: NaN values in similarity inputs, setting graph_sim to 0")
                graph_sim = torch.tensor(0.0, device=input_ids_source.device)
            else:
                graph_sim = torch.mean(cos(sagpooling_ouputs, mgat_outputs_summary))
        
        # Cross-entropy loss
        labels = output_ids[:, 1:].clone()
        
        # Validate labels more strictly
        labels = torch.clamp(labels, min=0, max=self.model.config.vocab_size-1)

        if self.args.label_smoothing == 0.0:
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            # Assuming you have this function defined elsewhere
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.args.label_smoothing, ignore_index=self.pad_token_id
            )
        
        # Validate loss values
        if torch.isnan(loss):
            print("ERROR: NaN loss detected!")
            print(f"lm_logits stats: min={lm_logits.min()}, max={lm_logits.max()}, mean={lm_logits.mean()}")
            print(f"labels stats: min={labels.min()}, max={labels.max()}")
            print(f"Has inf in lm_logits: {torch.isinf(lm_logits).any()}")
            raise ValueError("NaN loss detected")
        
        if torch.isnan(graph_sim):
            print("WARNING: NaN graph_sim detected, setting to 0")
            graph_sim = torch.tensor(0.0, device=input_ids_source.device)

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
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            use_cache=False,
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
        else:
            logs = {"vloss": vloss}
            self.logger.log_metrics(logs, step=self.global_step)
        
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
        precision=32,
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
        precision=32,
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
