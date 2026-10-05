import itertools
import numpy as np
from functools import partial
from tot.llm_wrapper import get_response

def extract_state(line):
    if 'left: ' in line:
        raw_nums = line.split('left: ')[-1].split(')')[0].strip().split()
        try:
            return tuple(sorted([float(n) for n in raw_nums]))
        except ValueError:
            return tuple(sorted(raw_nums))
    return line

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = get_response(value_prompt, n=n_evaluate_sample, stop=None, max_tokens=150)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = get_response(vote_prompt, n=n_evaluate_sample, stop=None, max_tokens=150)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    raw_responses = get_response(propose_prompt, n=1, stop=None, max_tokens=1000)
    raw_text = raw_responses[0] if raw_responses else ""
    proposals = [line.strip() for line in raw_text.split('\n') if line.strip()]
    if 'left: ' in raw_text:
        valid = [p for p in proposals if 'left: ' in p]
        if valid:
            seen_states = set()
            deduped = []
            for p in valid:
                st = extract_state(p)
                if st not in seen_states:
                    seen_states.add(st)
                    deduped.append(p)
            proposals = deduped
    return [y + _ + '\n' for _ in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = get_response(prompt, n=n_generate_sample, stop=stop, max_tokens=1000)
    return [y + _ for _ in samples]

def solve(args, task, idx, to_print=True):
    global get_response
    get_response = partial(get_response, model=args.backend, temperature=args.temperature)
    print(get_response)
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        if not new_ys:
            if to_print:
                print(f"Warning: No valid candidate paths generated at step {step}")
            break
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        # selection
        if args.method_select == 'sample':
            val_sum = sum(values)
            if val_sum > 0:
                ps = np.array(values) / val_sum
                select_ids = np.random.choice(ids, size=min(args.n_select_sample, len(ids)), p=ps).tolist()
            else:
                select_ids = ids[:args.n_select_sample]
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print: 
            if new_ys and values:
                sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            else:
                sorted_new_ys, sorted_values = (), ()
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global get_response
    get_response = partial(get_response, model=args.backend, temperature=args.temperature)
    print(get_response)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}
