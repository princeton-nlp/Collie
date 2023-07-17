import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import dill
import json
import tqdm
import argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
)

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

OPEN_MODELS = {
    'vicuna',
    'longchat',
    'wizard',
    'orca',
}

def print_gpu_usage():
    nvmlInit()
    n_gpus = torch.cuda.device_count()

    print('========== GPU Utilization ==========')
    for gpu_id in range(n_gpus):
        h = nvmlDeviceGetHandleByIndex(gpu_id)
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'GPU {gpu_id}')
        print(f'- Used:       {info.used / 1024 ** 3:>8.2f} B ({info.used / info.total * 100:.1f}%)')
        print(f'- Available:  {info.free / 1024 ** 3:>8.2f} B ({info.free / info.total * 100:.1f}%)')
        print(f'- Total:      {info.total / 1024 ** 3:>8.2f} B')
    print('=====================================')

def top_p_filtering(logits, top_p):
    """
    logits: (1, vocab_size)
    Code taken from: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    logits = logits.squeeze(0)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    return logits 

class OpenLM(nn.Module):
    def __init__(self, model_name, device_map='auto'):
        super().__init__()
        self.model_name = model_name
        self.device_map = device_map
        self.system_msg = "Below is a chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and accurate answers to the user's questions and follows the user's instruction exactly.\n\nUSER: "
        model, tokenizer, hf_model_name = self.get_lm_by_name(model_name, device_map)
        self.model = model
        self.tokenizer = tokenizer
        self.hf_model_name = hf_model_name
    @torch.inference_mode()
    def generate(
        self,
        text,
        max_new_tokens=500,
        temperature=0.7,
        top_p=0.92,
        show_gpu=False,
    ):
        if self.system_msg is not None:
            text = self.system_msg + text + '\n\nASSISTANT:'
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.model.device)
        info = dict()
        max_length_reached = True
        generated_ids = []
        for t in range(max_new_tokens):
            if t == 0:
                out = self.model(input_ids=input_ids, use_cache=True)
            else:
                out = self.model(
                    input_ids=generated_id.unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            logits = out.logits[:, -1, :]
            past_key_values = out.past_key_values
    
            logits = top_p_filtering(logits, top_p=top_p)
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            generated_id = torch.multinomial(probs, num_samples=1)
            generated_ids.append(generated_id.squeeze(0).item())
    
            if generated_id.item() == self.tokenizer.eos_token_id:
                max_length_reached = False
                break
        info['max_length_reached'] = max_length_reached
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return dict(
            generated_text=generated_text,
            info=info,
        )
    
    @staticmethod
    def get_lm_by_name(model_name, device_map):
        if 'vicuna' in model_name:
            hf_model_name = f'lmsys/{model_name}-v1.3'
            tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=False, legacy=False)
            model = AutoModelForCausalLM.from_pretrained(hf_model_name, device_map=device_map)
        elif 'longchat' in model_name:
            hf_model_name = f'lmsys/{model_name}-16k'
            tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=False, legacy=False)
            from fastchat.model.llama_condense_monkey_patch import replace_llama_with_condense
            replace_llama_with_condense(8)
        
            model = AutoModelForCausalLM.from_pretrained(hf_model_name, revision='main', low_cpu_mem_usage=True, device_map=device_map)
        elif 'wizard' in model_name:
            prefix, model_size = model_name.split('-')
            hf_model_name = f'WizardLM/{prefix.capitalize()}LM-{model_size.upper()}-V1.1'
            tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=False, legacy=False)
            model = AutoModelForCausalLM.from_pretrained(hf_model_name, device_map=device_map)
        elif 'orca' in model_name:
            prefix, model_size = model_name.split('-')
            hf_model_name = f'Open-Orca/Open{prefix.capitalize()}-Preview1-{model_size.upper()}'
            tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=False, legacy=False)
            model = AutoModelForCausalLM.from_pretrained(hf_model_name, device_map=device_map)
        elif model_name == 'alpaca-7b':
            hf_model_name = f'chavinlo/alpaca-native'  # hope it's good?
            tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=False, legacy=False)
            model = AutoModelForCausalLM.from_pretrained(hf_model_name, device_map=device_map)
        else:
            raise ValueError(f'Model name `{model_name}` not supported.')
        return model, tokenizer, hf_model_name


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, choices=['vicuna-7b', 'alpaca-7b'], required=True)
    args.add_argument('--id', type=int, default=0)
    args.add_argument('--N', type=int, default=5)
    args = args.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model = OpenLM(model_name=args.model)

    with open("data/all_data.dill", "rb") as f:
        all_data = dill.load(f)

    prompts = []
    for k in all_data:
        for example in all_data[k]:
            if example["prompt"] not in prompts:
                prompts.append(example["prompt"])

    print(len(prompts), args.N)
    prompts = prompts * args.N
    texts = []
    for prompt in tqdm.tqdm(prompts):
        out = model.generate(text=prompt, max_new_tokens=1000, show_gpu=True)
        generated_text = out['generated_text']
        texts.append(generated_text)
        if len(texts) % 100 == 0 or len(texts) == len(prompts):
            with open(f"logs/{args.model}-{args.N}trial-no{args.id}-prompt.json", "w") as f:
                json.dump({"prompts": prompts, "texts": texts}, f, indent=2)