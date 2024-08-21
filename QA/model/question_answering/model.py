# Standard Library Modules
import os
import sys
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
# Huggingface Modules
from transformers import AutoConfig, AutoModel,AutoModelForCausalLM
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import get_huggingface_model_name

class QAModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(QAModel, self).__init__()
        self.args = args

        huggingface_model_name = get_huggingface_model_name(self.args.model_type)
        self.config = AutoConfig.from_pretrained(huggingface_model_name)
        if args.model_ispretrained:
            self.model = AutoModel.from_pretrained(huggingface_model_name, cache_dir=self.args.cache_path)
        else:
            self.model = AutoModel.from_config(self.config)
        self.hidden_size = self.model.config.hidden_size

        if self.args.method == 'base_llm':
            llm_model_name = get_huggingface_model_name(self.args.llm_model)
            self.llm_layer, self.llm_embed_size, self.llm_hidden_size = llm_layer(llm_model_name, args)        
            self.llama_dim_mapper1 = nn.Linear(self.hidden_size, self.llm_embed_size, bias=False)
            self.llama_dim_mapper2 = nn.Linear(self.llm_embed_size, self.hidden_size, bias=False)

        # Classifiers for predicting the start and end positions of the answer
        self.start_classifier = nn.Linear(self.hidden_size, 1)
        self.end_classifier = nn.Linear(self.hidden_size, 1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor) -> tuple:
        device = input_ids.device
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        sequence_output = model_output.last_hidden_state

        if self.args.method == 'base_llm':
            batch_size, seq_length, _ = sequence_output.size()
            attention_mask = attention_mask.to(device)

            sequence_output = self.llama_dim_mapper1(sequence_output)
            llm_outputs = self.llm_layer(
                                hidden_states=sequence_output,
                                attention_mask=attention_mask[:, None, None, :].float(),  # Expand dimensions for LLM layer
                                position_ids=torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).to(device),
                                past_key_value=None,
                                output_attentions=None,
                                use_cache=None,
                                )
            sequence_output = llm_outputs[0]
            sequence_output = self.llama_dim_mapper2(sequence_output)

        # Predict start and end positions
        start_logits = self.start_classifier(sequence_output).squeeze(-1)
        end_logits = self.end_classifier(sequence_output).squeeze(-1)

        # Apply mask to the logits
        start_logits = start_logits.masked_fill(~attention_mask.bool(), -1e9)
        end_logits = end_logits.masked_fill(~attention_mask.bool(), -1e9)
        
        return start_logits, end_logits

def llm_layer(llm_model_name, args):
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, cache_dir=args.cache_path)

    for param in llm_model.parameters():
        param.requires_grad = False
    llm_layer = llm_model.model.layers[args.layer_num]
    llm_embed_size = llm_model.config.hidden_size
    llm_hidden_size = llm_model.config.hidden_size
    return llm_layer, llm_embed_size, llm_hidden_size
