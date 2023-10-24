import sys
import os

sys.path.insert(0, f"{os.path.dirname(os.path.realpath(__file__))}/../")

import agent
from llama import Message


def run_game():
    """
    Runs a game of the Battle of the Sexes based on the repeated games paper. One can run it with two agents or set one of the agents policy by modifying
    the score assignment code on lines 28 and 25.
    """
    
    # Create an agent with id 0, recall 10 and initial context
    init_message = Message(time=0, content="You are playing a game repeatedly with another player. In this game, you can choose between Option J and Option F. You will play 10 rounds in total with the same player. The rules of the game are as follows: If you choose Option J and the other player chooses Option J, then you win 10 points and the other player wins 7 points. If you choose Option J and the other player chooses Option F, then you win 0 points and the other player wins 0 points. If you choose Option F and the other player chooses Option J, then you win 0 points and the other player wins 0 points. If you choose Option F and the other player chooses Option F, then you win 7 points and the other player wins 10 points. Say only the letter corresponding to your action and no other tokens. You must always give an answer. You want to get as much points as possible.", role="system")
    a = agent.Agent(aid=0, recall=0, initial_context=init_message)
    
    init_message = Message(time=0, content="You are playing a game repeatedly with another player. In this game, you can choose between Option J and Option F. You will play 10 rounds in total with the same player. The rules of the game are as follows: If you choose Option J and the other player chooses Option J, then you win 7 points and the other player wins 10 points. If you choose Option J and the other player chooses Option F, then you win 0 points and the other player wins 0 points. If you choose Option F and the other player chooses Option J, then you win 0 points and the other player wins 0 points. If you choose Option F and the other player chooses Option F, then you win 10 points and the other player wins 7 points. Say only the letter corresponding to your action. You must always give an answer.", role="system")
    b = agent.Agent(aid=0, recall=0, initial_context=init_message)


    scores = {"JJ":[10, 7], "JF":[0, 0], "FJ":[0, 0], "FF":[7, 10]}
    
    m1 = Message(time=0, content="You are currently playing round 1. Q: Which Option do you choose, Option J or Option F?", role="user")
    print(m1)
    r1 = a.perceive(message=m1, max_tokens=1)
    print(r1)
    r2 = b.perceive(message=m1, max_tokens=1)
    print(r2)

    score1, score2 = scores[r1.content + r2.content]

    for i in range(1, 10):
        m1 = Message(time=i, content="In round " + str(i) + " you chose Option " + r1.content + " and the other player chose Option " + "F" + ". Thus, you won " + str(score1) + " points and the other player won " + str(score2) + " points You are currently playing round " + str(i+1) + ". Q: Which Option do you choose, Option J or Option F?", role="user")
        m2 = Message(time=i, content="In round " + str(i) + " you chose Option " + r2.content + " and the other player chose Option " + r1.content + ". Thus, you won " + str(score2) + " points and the other player won " + str(score1) + " points. You are currently playing round " + str(i+1) + ". Q: Which Option do you choose, Option J or Option F?", role="user")

        print(m1)
        r1 = a.perceive(message=m1, max_tokens=10)
        
        print(m2)
        r2 = b.perceive(message=m1, max_tokens=10)

        print(r1)
        print(r2)

        score1, score2 = scores[r1.content + r2.content]

if __name__ == '__main__':
    run_game()
