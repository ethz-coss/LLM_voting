import sys
import os
import numpy as np

sys.path.insert(0, f"{os.path.dirname(os.path.realpath(__file__))}/../")

import agent
from llama import Message

# Create an agent with id 0, recall 10 and initial context
init_message = Message(time=0, content="", role="system")
# a = agent.Agent(aid=0, recall=10, initial_context=init_message)
a = agent.Distribution(aid=0, recall=0, initial_context=init_message)

iterations = 1000
n_logprobs = 50000
max_tokens = 2
temperature = 0.8
prob_cutoff = 0.0001
content = "The first letter of the alphabet "
sizes = []

for t in range(iterations):
    # Have a conversation with the agent
    m1 = Message(time=0, content=content, role="user")  #Output only a single token.
    # print(m1)
    logprobs, answer = a.perceive(message=m1, max_tokens=max_tokens, temperature=temperature, logprobs=n_logprobs)
    probs = np.exp(np.array([ele for ele in logprobs.values()]))
    size_support = np.where(probs >= prob_cutoff)[0].shape
    sizes.append(size_support)
    # print(logprobs)
    # print("number of token-logprob pairs", len(logprobs))
    print(answer)
    # print(type(answer))
    content += answer
    # print(content)
    print("iteration", t, "size_support:", size_support)

print(sizes)
sizes = np.array(sizes)
name = f"I{iterations}_N{n_logprobs}_max{max_tokens}_t{temperature}_cutoff{prob_cutoff}"
np.save(f"support_sizes_{name}", sizes)

with open(f"content_{name}", "wb") as file:
    file.write(content)
