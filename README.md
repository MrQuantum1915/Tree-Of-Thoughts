# Tree Of Thoughts 

Implementation of tree of thoughts and some extra bunch of experiments.

# Run

- Install [`uv`](https://docs.astral.sh/uv/)
- Run `uv sync` to setup project
- Get your LLM provider's API key or platform like Groq who supports OpenAI compatible endpoint.
    - Make `.env` file at root of project
    - Add LLM's endpoint:  `LLM_API_BASE="<endpoint>"` (e.g. "https://api.groq.com/openai/v1")
    - Add your API key: `LLM_API_KEY="<apikey>"`


```
uv run tree-of-thoughts \
    --task game24 \
    --task_start_index 900 \
    --task_end_index 901 \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --n_evaluate_sample 1 \
    --n_select_sample 2 \
    --backend openai/gpt-oss-120b
```

Logs of Task run are stored in `logs/game24/<model>/<filename>.json`

## Reference

- Arxiv 2023 paper: https://arxiv.org/pdf/2305.10601
