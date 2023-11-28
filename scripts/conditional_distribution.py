import sys
import os
import numpy as np

sys.path.insert(0, f"{os.path.dirname(os.path.realpath(__file__))}/../")

import agent
from llama import Message

iterations = 10
n_logprobs = 5
max_tokens = 1
temperature = 0.0
prob_cutoff = 0.0001
content = "The first letter of the alphabet"
sizes = []
time = 1
total_prob = 1

init_message = Message(time=0, content="", role="system")
a = agent.Distribution(aid=0, recall=0, initial_context=init_message, temperature=temperature)


for t in range(iterations):
    # Have a conversation with the agent
    m1 = Message(time=time, content=content, role="user")  #Output only a single token.
    answer, logprobs = a.perceive(message=m1, max_tokens=max_tokens, logprobs=n_logprobs)
    probs = np.exp(np.array([ele for ele in logprobs[0].values()]))

    if answer.content == "\n":
        break

    print(probs)
    total_prob *= probs[0]
    
    # size_support = np.where(probs >= prob_cutoff)[0].shape
    # sizes.append(size_support)
    # print(logprobs)
    # print("number of token-logprob pairs", len(logprobs))
    # print(type(answer))

    content += answer.content
    time+=1
    # print(content)
    # print("iteration", t, "size_support:", size_support)

print(content)
print(total_prob)
    
# print(sizes)
# sizes = np.array(sizes)
# name = f"I{iterations}_N{n_logprobs}_max{max_tokens}_t{temperature}_cutoff{prob_cutoff}"
# np.save(f"support_sizes_{name}", sizes)

# with open(f"content_{name}", "wb") as file:
#     file.write(content)
