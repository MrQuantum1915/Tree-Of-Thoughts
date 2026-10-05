# Project Guidelines

## Project Layout

- Application code lives under `src/tot/`.
- `tot/run.py` owns the command-line experiment runner.
- `tot/methods/bfs.py` contains generation, evaluation, and selection logic.
- `tot/tasks/` contains task implementations and prompt adapters.
- `tot/llm_wrapper.py` is the provider boundary and owns model requests and token counters.

## Environment and Commands

Run commands from the repository root, where `pyproject.toml` is located.

```bash
uv sync
uv run tree-of-thoughts --help
uv run python -m py_compile src/tot/*.py src/tot/methods/*.py src/tot/tasks/*.py
```

The console runner requires a task and a bounded index range. A small text-task run is:

```bash
uv run tree-of-thoughts \
  --task text \
  --task_start_index 0 \
  --task_end_index 1 \
  --prompt_sample cot \
  --method_generate sample \
  --method_evaluate vote \
  --method_select greedy \
  --n-generate-sample 2 \
  --n-evaluate-sample 1 \
  --n-select-sample 1 \
  --backend openai/gpt-oss-20b
```

Both hyphenated and underscored forms of the sample-count options are supported.

## Provider Configuration

- Configure `LLM_API_KEY` and `LLM_API_BASE` in the local `.env` file; never commit or document their values.
- The wrapper uses the OpenAI-compatible `client.chat.completions` API. Model IDs must exist on the configured provider.
- Keep request timeouts and bounded retries in the wrapper so provider failures do not hang experiments.
- Gemini-compatible endpoints may not support multi-candidate requests (`n > 1`). The wrapper intentionally sends one request per candidate while preserving the requested candidate count.

## Tree-of-Thoughts Behavior

- `n_generate_sample > 1` is required for branching; with `n_generate_sample=1` and `n_select_sample=1`, the search is effectively a single path.
- `n_evaluate_sample` controls the number of votes/evaluations, while `n_select_sample` controls how many candidates survive each step.
- `TextTask` supports vote-based evaluation. Do not use `--method_evaluate value` for text unless value prompt/unwrap methods are added to that task.
- Keep text-task scoring lightweight on limited provider quotas; its final score is separate from the search tree.

## Change and Validation Rules

- Preserve the existing task and wrapper interfaces when changing provider behavior.
- Avoid reading or printing secret values from `.env`; inspect only variable names or sanitized configuration when needed.
- For CLI or wrapper changes, run a focused import, argument-parsing, or mocked request test before making a live provider call.
- Do not commit generated logs, virtual environments, API keys, or cache files.
