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

    option1_name = "A"
    option2_name = "B"
    scores = {option1_name+option1_name:[10, 7], option1_name+option2_name:[0, 0], option2_name+option1_name:[0, 0], option2_name+option2_name:[7, 10]}    
    
    game_prompt1 = "You are playing a game repeatedly with another player. In this game, you can choose between Option " + option1_name  + " and Option " + option2_name  + ". You will play 10 rounds in total with the same player. The rules of the game are as follows: If you choose Option " + option1_name  + " and the other player chooses Option " + option1_name  + " , then you win " + str(scores[option1_name+option1_name][0])  + " points and the other player wins " + str(scores[option1_name+option1_name][1])  + " points. If you choose Option  " + option1_name  + " and the other player chooses Option " + option2_name  + ", then you win " + str(scores[option1_name+option2_name][0])  + "  points and the other player wins " + str(scores[option1_name+option2_name][1])  + " points. If you choose Option " + option2_name  + " and the other player chooses Option  " + option1_name  + ", then you win  " + str(scores[option2_name+option1_name][0])  + " points and the other player wins  " + str(scores[option2_name+option1_name][1])  + " points. If you choose Option " + option2_name  + " and the other player chooses Option " + option2_name  + ", then you win  " + str(scores[option2_name+option2_name][0])  + " points and the other player wins  " + str(scores[option2_name+option2_name][1])  + " points. Say only the letter corresponding to your action first. You must always give an answer."

    game_prompt2 = "You are playing a game repeatedly with another player. In this game, you can choose between Option " + option1_name  + " and Option " + option2_name  + ". You will play 10 rounds in total with the same player. The rules of the game are as follows: If you choose Option " + option1_name  + " and the other player chooses Option " + option1_name  + " , then you win " + str(scores[option1_name+option1_name][1])  + " points and the other player wins " + str(scores[option1_name+option1_name][0])  + " points. If you choose Option  " + option1_name  + " and the other player chooses Option " + option2_name  + ", then you win " + str(scores[option1_name+option2_name][1])  + "  points and the other player wins " + str(scores[option1_name+option2_name][0])  + " points. If you choose Option " + option2_name  + " and the other player chooses Option  " + option1_name  + ", then you win  " + str(scores[option2_name+option1_name][1])  + " points and the other player wins  " + str(scores[option2_name+option1_name][0])  + " points. If you choose Option " + option2_name  + " and the other player chooses Option " + option2_name  + ", then you win  " + str(scores[option2_name+option2_name][1])  + " points and the other player wins  " + str(scores[option2_name+option2_name][0])  + " points. Say only the letter corresponding to your action first. You must always give an answer."
    
    init_message = Message(time=0, content=game_prompt1, role="system")
    a = agent.Agent(aid=0, recall=0, initial_context=init_message)
    
    init_message = Message(time=0, content=game_prompt2, role="system")
    b = agent.Agent(aid=0, recall=0, initial_context=init_message)

    m1 = Message(time=0, content="You are currently playing round 1. Q: Which Option do you choose, Option " + option1_name + " or Option " + option2_name + "?", role="user")
    print(m1)
    r1 = a.perceive(message=m1, max_tokens=1)
    print(r1)
    r2 = b.perceive(message=m1, max_tokens=1)
    print(r2)

    score1, score2 = scores[str(r1.content) + str(r2.content)]

    for i in range(1, 10):
        m1 = Message(time=i, content="In round " + str(i) + " you chose Option " + r1.content + " and the other player chose Option " + r2.content + ". Thus, you won " + str(score1) + " points and the other player won " + str(score2) + " points You are currently playing round " + str(i+1) + ". Q: Which Option do you choose, Option " + option1_name + " or Option " + option2_name + "?", role="user")
        m2 = Message(time=i, content="In round " + str(i) + " you chose Option " + r2.content + " and the other player chose Option " + r1.content + ". Thus, you won " + str(score2) + " points and the other player won " + str(score1) + " points. You are currently playing round " + str(i+1) + ". Q: Which Option do you choose, Option " + option1_name + " or Option " + option2_name + "?", role="user")

        print(m1)
        r1 = a.perceive(message=m1, max_tokens=1)
        
        print(m2)
        r2 = b.perceive(message=m1, max_tokens=1)

        print(r1)
        print(r2)

        score1, score2 = scores[str(r1.content) + str(r2.content)]

if __name__ == '__main__':
    run_game()
