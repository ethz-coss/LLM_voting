import sys
import os

sys.path.insert(0, f"{os.path.dirname(os.path.realpath(__file__))}/../")

from typing import List, Tuple
import pandas as pd
import agent
from llama import Message
import yaml

class colors: # You may need to change color settings
    RED = '\033[31m'
    ENDC = '\033[m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'

def agent_initial_context(name: str):
    file_path = os.path.join(os.path.dirname(__file__), 'persona.yml')
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    description = data[name]['learned']
    description = description + 'You are short in you replies. You never use more than 100 words.'
    description_str = str(description)
    return Message(time=0, content=description_str, role="system")


def run_two_agent_conversation(n_steps: int = 10, initial_context_a1: Message = None, initial_context_a2: Message = None, 
                               trigger_sentence: Message = None, max_tokens: int = 300, temperature: float = 0.8) -> List[Tuple[int, Message]]:
    initial_context_a1 = initial_context_a1 if initial_context_a1 is not None else Message(time=0, content="Your name is John. You are a student from Atlanta. You communicate in a terse fashion only saying the minimum necessary. Please impersonate John for the time being and answer all questions you might be asked. If you do not know the answer please say so. You remember the following interactions.", role="system")
    initial_context_a2 = initial_context_a2 if initial_context_a2 is not None else Message(time=0, content="Your name is Carla. You are a student from New York. You love ballet. You communicate in a terse fashion only saying the minimum necessary. Please impersonate Carla for the time being and answer all questions you might be asked. If you do not know the answer please say so. You remember the following interactions.", role="system")
    trigger_sentence = trigger_sentence if trigger_sentence is not None else Message(time=1, content="Hi!", role="user")
    agent1 = agent.Agent(aid=0, recall=10, initial_context=initial_context_a1)
    agent2 = agent.Agent(aid=1, recall=10, initial_context=initial_context_a2)

    messages = [[1, trigger_sentence]]
    c = [colors.RED, colors.BLUE]
    for i in range(n_steps):
        current_message = messages[-1][1]
        current_message.role = 'user'
        if i % 2 == 0:
            current_message = agent1.perceive(message=current_message, max_tokens=max_tokens, temperature=temperature)
            messages.append([agent1.id, current_message])
        else:
            current_message = agent2.perceive(message=current_message, max_tokens=max_tokens, temperature=temperature)
            messages.append([agent2.id, current_message])

        print(f'Step {i} --- {c[i%2]}{messages[-1][0]} {messages[-1][1].content}{colors.ENDC}')

    return messages


if __name__ == '__main__':
    messages = run_two_agent_conversation(n_steps=100, max_tokens=100, temperature=0)
