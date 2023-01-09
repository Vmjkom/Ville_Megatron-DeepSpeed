import argparse
from transformers.models.megatron_bert.convert_megatron_bert_checkpoint import convert_megatron_checkpoint
from transformers import GPT2Config,BertConfig
import os
import torch
import json


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default=None, type=str, help='Input Checkpoint folder')
    parser.add_argument('--output_folder', default=None, type=str, help='Output hf checkpoint folder')
    parser.add_argument('--for_release', action='store_true', help='Convert for release purpose, reset some (progress) counters.')
    args = parser.parse_args()
    print(f'args = {args}')
    return args
    
def main():


    args = parse_arguments()
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
    output_config["architectures"] = ["GPT2LMHeadModel"]
    output_config["model_type"] = "gpt2"
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