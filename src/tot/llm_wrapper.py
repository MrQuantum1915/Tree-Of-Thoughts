import os
import openai
import backoff 
from dotenv import load_dotenv

DEFAULT_MODEL = "openai/gpt-oss-20b"
SYSTEM_PROMPT = (
    "You are a concise assistant solving reasoning tasks. "
    "Follow the exact output format of the examples. "
    "Do not include markdown tables, explanations, conversational text, or bolding."
)

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

@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError),
    max_time=300,
)
def completions_with_backoff(**kwargs):
    client = openai.OpenAI(
        api_key=api_key or None,
        base_url=api_base or None,
        max_retries=0,
        timeout=60.0,
    )
    request_kwargs = {key: value for key, value in kwargs.items() if value is not None}
    model = request_kwargs.get("model", "")
    if "gpt-oss" in model:
        request_kwargs.setdefault("reasoning_effort", "low")
    return client.chat.completions.create(**request_kwargs)

def get_response(prompt, model=DEFAULT_MODEL, temperature=0.7, max_tokens=350, n=1, stop=None, system_prompt=SYSTEM_PROMPT) -> list:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return response_handler(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def response_handler(messages, model=DEFAULT_MODEL, temperature=0.7, max_tokens=350, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        n -= 1
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=1, stop=stop)
        for choice in res.choices:
            content = choice.message.content or ""
            outputs.append(content)
        # log completion tokens
        if res.usage:
            completion_tokens += res.usage.completion_tokens or 0
            prompt_tokens += res.usage.prompt_tokens or 0
    return outputs

def llm_usage(backend=DEFAULT_MODEL):
    # based on groq's inference pricing
    global completion_tokens, prompt_tokens
    cost = 0.0
    if backend == "openai/gpt-oss-20b":
        cost = completion_tokens / 1e6 * 0.3 + prompt_tokens / 1e6 * 0.075
    elif backend == "openai/gpt-oss-120b":
        cost = completion_tokens / 1e6 * 0.6 + prompt_tokens / 1e6 * 0.15
    elif backend == "qwen/qwen3.8-27b":
        cost = completion_tokens / 1e6 * 4.0 + prompt_tokens / 1e6 * 0.8
    else:
        cost = completion_tokens / 1e6 * 0.5 + prompt_tokens / 1e6 * 0.1
    
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
