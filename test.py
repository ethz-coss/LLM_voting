import agent
import numpy as np
from llama import Message
from copy import deepcopy


def test():
    """
    This is a test function for the agent module. It illustrates how to have a conversation with the agent.
    """

    # Create an agent with id 0, recall 10 and initial context
    init_message = Message(time=0, content="You are a helpful assistant.", role="system")
    # a = agent.Agent(aid=0, recall=10, initial_context=init_message)
    a = agent.Agent(aid=0, recall=10, initial_context=init_message, temperature=0.8)    
    m0 = Message(time=0, content="What is the capital of Poland?", role="user")

    ans0 = a.perceive(message=m0, max_tokens=1000)
    print(ans0)

    
    # Have a conversation with the agent
    # m1 = Message(time=1, content="Is the following sentence grammatical: \"The old  man the boat.\"?", role="user")  #Output only a single token.
    # print(m1)

    # ans1 = a.perceive(message=m1, max_tokens=1000, temperature=0)
    # print(ans1)

    
    # for token_probs in logprobs1:
    #     print([(k, np.exp(token_probs[k])) for k in token_probs.keys()])

    # m2 = Message(time=2, content="Cool to meet you John. I am Carla. I am 25 years old. I study architecture.", role="user")
    # print(m2)
    # r2 = a.perceive(message=m2, max_tokens=60)
    # print(r2)

    # m3 = Message(time=3, content="What do you do?", role="user")
    # print(m3)
    # r3 = a.perceive(message=m3, max_tokens=60)
    # print(r3)

    # # This is a message to test if the agent remembers past messages
    # m4 = Message(time=4, content="How old is Carla and what does she do?", role="user")
    # print(m4)
    # r4 = a.perceive(message=m4, max_tokens=60)
    # print(r4)


if __name__ == '__main__':
    test()
