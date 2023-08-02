import os
import re
import openai
import time
import random
import aiolimiter
import asyncio
import logging
import tenacity
import google.generativeai as palm
from aiohttp import ClientSession
from typing import Any, List, Dict, Union
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from google.api_core import retry, exceptions
from litellm import completion


# OpenAI GPT with ChatCompletion
completion_tokens = {"gpt-4": 0, "gpt-3.5-turbo": 0}
prompt_tokens = {"gpt-4": 0, "gpt-3.5-turbo": 0}

async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop: Union[str, List[str]],
    limiter: aiolimiter.AsyncLimiter,
) -> Dict[str, Any]:
    async with limiter:
        for _ in range(10000000000):
            try:
                return await openai.ChatCompletion.acreate(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop,
                )
            except openai.error.OpenAIError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
    messages_list: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop: Union[str, List[str]],
    requests_per_minute: int = 300,
) -> List[str]:
    if model == "gpt-4":
        requests_per_minute = 200
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ["OPENAI_API_KEY"]
    session = ClientSession()
    openai.aiosession.set(session)
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            limiter=limiter,
        )
        for messages in messages_list
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    await session.close()
    # return [x["choices"][0]["message"]["content"] for x in responses]
    return responses


def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    return gpts([prompt] * n, model=model, temperature=temperature, max_tokens=max_tokens, stop=stop)

def gpts(prompts, model="gpt-4", temperature=0.7, max_tokens=1000, stop=None) -> list:
    messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
    return chatgpts(messages_list, model=model, temperature=temperature, max_tokens=max_tokens, stop=stop)

def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    return chatgpts([messages] * n, model=model, temperature=temperature, max_tokens=max_tokens, stop=stop)

def chatgpts(messages_list, model="gpt-4", temperature=0.7, max_tokens=1000, stop=None) -> list:
    responses =  asyncio.run(generate_from_openai_chat_completion(model=model, messages_list=messages_list, temperature=temperature, max_tokens=max_tokens, top_p=1, stop=stop))
    texts = [x["choices"][0]["message"]["content"] for x in responses]
    # print(responses)
    global completion_tokens, prompt_tokens
    completion_tokens[model] += sum(x["usage"]["completion_tokens"] for x in responses if "usage" in x and "completion_tokens" in x["usage"])
    prompt_tokens[model] += sum(x["usage"]["prompt_tokens"] for x in responses if "usage" in x and "prompt_tokens" in x["usage"])
    return texts

def gpt_usage():
    global completion_tokens, prompt_tokens
    cost = completion_tokens["gpt-4"] / 1000 * 0.06 + prompt_tokens["gpt-4"] / 1000 * 0.03
    cost += completion_tokens["gpt-3.5-turbo"] / 1000 * 0.002 + prompt_tokens["gpt-3.5-turbo"] / 1000 * 0.0015
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}



# Google PaLM with TextGeneration
palm.configure(api_key=os.environ["PALM_API_KEY"])


@retry.Retry()
def retry_chat(**kwargs):
    return palm.chat(**kwargs)

@retry.Retry()
def retry_reply(x, arg):
    return x.reply(arg)

@tenacity.retry(wait=tenacity.wait_random_exponential(min=3, max=60), stop=tenacity.stop_after_attempt(6))
def generate_text(*args, **kwargs):
    return palm.generate_text(*args, **kwargs)

def palm_llm(prompt, model="models/text-bison-001", n=1, temperature=0.7):
    # Request configuration disabling all safety settings to prevent blocking
    config = {
        'model': model,
        'temperature': temperature,
        'top_p': 1,
        'candidate_count': n,
        'safety_settings': [
            {"category":"HARM_CATEGORY_DEROGATORY","threshold":1},
            {"category":"HARM_CATEGORY_TOXICITY","threshold":1},
            {"category":"HARM_CATEGORY_VIOLENCE","threshold":1},
            {"category":"HARM_CATEGORY_SEXUAL","threshold":1},
            {"category":"HARM_CATEGORY_MEDICAL","threshold":1},
            {"category":"HARM_CATEGORY_DANGEROUS","threshold":1}
        ],
    }
    # to prevent rate limiting
    time.sleep(2)
    try:
        response = generate_text(
            **config,
            prompt=prompt
        )
    except Exception as e:
        print(e)
        return ""
    output = response.result
    if isinstance(output, str):
        output = re.sub("\n", " ", output)
    return output

def palm_llms(prompts, model="models/text-bison-001", temperature=0.7):
    return [palm_llm(prompt, model=model, n=1, temperature=temperature) for prompt in tqdm(prompts)]

# Overall
def llms(prompts, model, temperature=0.7, max_tokens=1000, stop=None) -> list:
    messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
    if model.startswith("palm"):
        model = "text-bison-001"
    return completion(model=model, messages=messages_list, max_tokens=max_tokens, temperature=temperature, stop=stop)["choices"][0]["message"]["content"]
    else:
        raise ValueError("Invalid model name.", model)