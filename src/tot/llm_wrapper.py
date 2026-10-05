import os
import openai
import backoff 
from dotenv import load_dotenv

DEFAULT_MODEL = "openai/gpt-oss-20b"

completion_tokens = prompt_tokens = 0

load_dotenv()

api_key = os.getenv("LLM_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: LLM_API_KEY is not set")
    
api_base = os.getenv("LLM_API_BASE", "")
if api_base != "":
    print("Warning: LLM_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

@backoff.on_exception(backoff.expo, openai.OpenAIError, max_time=10)
def completions_with_backoff(**kwargs):
    client = openai.OpenAI(
        api_key=api_key or None,
        base_url=api_base or None,
        max_retries=0,
        timeout=30.0,
    )
    request_kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return client.chat.completions.create(**request_kwargs)

def get_response(prompt, model=DEFAULT_MODEL, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return response_handler(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def response_handler(messages, model=DEFAULT_MODEL, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        n -= 1
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=1, stop=stop)
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs

def llm_usage(backend=DEFAULT_MODEL):
    # based on groq's inference pricing
    global completion_tokens, prompt_tokens
    
    if backend == "openai/gpt-oss-20b":
        cost = completion_tokens / 1e6 * 0.3 + prompt_tokens / 1e6 * 0.075

    if backend == "openai/gpt-oss-120b":
        cost = completion_tokens / 1e6 * 0.6 + prompt_tokens / 1e6 * 0.15
    
    if backend == "qwen/qwen3.8-27b":
        cost = completion_tokens / 1e6 * 4.0 + prompt_tokens / 1e6 * 0.8
    
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
