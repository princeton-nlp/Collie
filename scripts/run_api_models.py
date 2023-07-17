import dill
import json
import logging
import os
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from collie.models import llms, gpt_usage
from tqdm import tqdm
logging.getLogger().setLevel(logging.ERROR)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'palm-text-bison-001'], required=True)
    args.add_argument('--N', type=int, default=20)
    args = args.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    model, N = args.model, args.N

    # load all data
    with open("data/all_data.dill", "rb") as f:
        all_data = dill.load(f)

    # collect all prompts
    prompts = []
    for k in all_data:
        for example in all_data[k]:
            if example["prompt"] not in prompts:
                prompts.append(example["prompt"])

    prompts = prompts * N
    print(len(prompts))


    # chunk prompts by 100 pieces to call
    texts = []
    for x in tqdm(range(0, len(prompts), 100)):
        prompts_chunk = prompts[x:x+100]
        text_chunk = llms(prompts_chunk, model=model, temperature=0.7, max_tokens=1000, stop=None)
        texts.extend(text_chunk)
        with open(f"logs/{model}-{N}trial-prompt.json", "w") as f:
            json.dump({"prompts": prompts, "texts": texts}, f, indent=2)


    print(len(prompts), len(texts))
    print(gpt_usage())

