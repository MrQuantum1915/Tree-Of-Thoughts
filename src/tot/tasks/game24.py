import re
import os
import sympy
import pandas as pd
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.game24 import * 


def get_current_numbers(y: str) -> str:
    lines = [line.strip() for line in y.strip().split('\n') if line.strip()]
    if not lines:
        return ""
    last_line = lines[-1]
    if 'left: ' in last_line:
        raw_nums = last_line.split('left: ')[-1].split(')')[0].strip().split()
        try:
            sorted_nums = sorted(raw_nums, key=lambda n: float(n))
            return " ".join(sorted_nums)
        except ValueError:
            return " ".join(sorted(raw_nums))
    return ""


class Game24Task(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, file='24.csv'):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        path = os.path.join(DATA_PATH, '24', file)
        self.data = list(pd.read_csv(path)['Puzzles'])
        self.value_cache = {}
        self.steps = 4
        self.stops = ['\n'] * 4

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output(self, idx: int, output: str):
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        # normalize unicode math operators
        expression = expression.replace('×', '*').replace('÷', '/').replace('–', '-').replace('−', '-')
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}
        try:
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception as e:
            return {'r': 0}
            
    @staticmethod
    def standard_prompt_wrap(x: str, y: str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str='') -> str:
        return cot_prompt.format(input=x) + y
    
    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        current_numbers = get_current_numbers(y) if y.strip() else x.strip()
        if not current_numbers:
            current_numbers = x.strip()
        if current_numbers == '24':
            prompt = cot_prompt.format(input=x) + 'Steps:\n' + y
        else:
            prompt = propose_prompt.format(input=current_numbers)
        return prompt
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        lines = [line.strip() for line in y.strip().split('\n') if line.strip()]
        last_line = lines[-1] if lines else ''
        if 'left: ' not in last_line:  # last step
            ans = last_line.lower().replace('answer: ', '')
            return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        if not current_numbers:
            current_numbers = x.strip()
        return value_prompt.format(input=current_numbers)
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        value_names = []
        for output in value_outputs:
            lines = [l.strip().lower() for l in output.strip().split('\n') if l.strip()]
            val = 'impossible'
            for line in reversed(lines):
                if 'sure' in line:
                    val = 'sure'
                    break
                elif 'likely' in line:
                    val = 'likely'
                    break
                elif 'impossible' in line:
                    val = 'impossible'
                    break
            value_names.append(val)
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value