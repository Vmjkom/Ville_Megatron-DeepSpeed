#!/usr/bin/env python

import os
import torch
import json

from deepspeed_checkpoint import DeepSpeedCheckpoint
from deepspeed_to_megatron import _create_rank_checkpoint, parse_arguments

# the import was tested to work with this version
# https://github.com/huggingface/transformers/commit/0af901e83 if it diverges we may consider
# copying that version here instead
from transformers.models.megatron_bert.convert_megatron_bert_checkpoint import convert_megatron_checkpoint
#from transformers.models.megatron_gpt2.convert_megatron_gpt2_checkpoint import convert_megatron_checkpoint
from transformers import GPT2Config,BertConfig

def main():

    # this first part comes mainly from deepspeed_to_megatron.main
    args = parse_arguments()
    print(f'Converting DeepSpeed checkpoint in {args.input_folder} to HF Transformers checkpoint in {args.output_folder}')

    
    iteration = 1000000
    input_state_dict = args.input_folder

    # the 2nd part comes from transformers.models.megatron_gpt2.convert_megatron_gpt2_checkpoint.main
    # Spell out all parameters in case the defaults change.
    config = BertConfig(
        vocab_size=50105,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-05,
        pad_token_id=0,
    )

    # Convert.
    print("Converting to HF Checkpoint")
    output_state_dict = convert_megatron_checkpoint(args, input_state_dict, config)

    basename = args.output_folder
    os.makedirs(basename, exist_ok=True)

    # Print the structure of converted state dict.
    #if args.print_checkpoint_structure:
    #    recursive_print(None, output_state_dict)

    # Store the config to file.
    output_config_file = os.path.join(basename, "config.json")
    output_config = config.to_dict()
    output_config["architectures"] = ["BertForMaskedLM"]
    output_config["model_type"] = "bert"
    print(f'Saving config to "{output_config_file}"')
    with open(output_config_file, "w") as f:
        json.dump(output_config, f)

    # Store the state_dict to file.
    output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)

    print("Now add tokenizer files and upload to the hub")


if __name__ == "__main__":
    main()
