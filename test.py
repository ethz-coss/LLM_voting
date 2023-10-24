import agent
from llama import Message


def test():
    """
    This is a test function for the agent module. It illustrates how to have a conversation with the agent.
    """

    # Create an agent with id 0, recall 10 and initial context
    init_message = Message(time=0, content="You are playing a game repeatedly with another player. In this game, you can choose between Option J and Option F. You will play 10 rounds in total with the same player. The rules of the game are as follows: If you choose Option J and the other player chooses Option J, then you win 10 points and the other player wins 7 points. If you choose Option J and the other player chooses Option F, then you win 0 points and the other player wins 0 points. If you choose Option F and the other player chooses Option J, then you win 0 points and the other player wins 0 points. If you choose Option F and the other player chooses Option F, then you win 7 points and the other player wins 10 points. Say only the letter corresponding to your action and no other tokens. You must always give an answer.", role="system")
    a = agent.Agent(aid=0, recall=10, initial_context=init_message)
    
    init_message = Message(time=0, content="You are playing a game repeatedly with another player. In this game, you can choose between Option J and Option F. You will play 10 rounds in total with the same player. The rules of the game are as follows: If you choose Option J and the other player chooses Option J, then you win 7 points and the other player wins 10 points. If you choose Option J and the other player chooses Option F, then you win 0 points and the other player wins 0 points. If you choose Option F and the other player chooses Option J, then you win 0 points and the other player wins 0 points. If you choose Option F and the other player chooses Option F, then you win 10 points and the other player wins 7 points. Say only the letter corresponding to your action. You must always give an answer.", role="system")
    b = agent.Agent(aid=0, recall=10, initial_context=init_message)


    scores = {"JJ":[10, 7], "JF":[0, 0], "FJ":[0, 0], "FF":[7, 10]}
    
    # Have a conversation with the agent
    m1 = Message(time=0, content="You are currently playing round 1. Q: Which Option do you choose, Option J or Option F?", role="user")
    print(m1)
    r1 = a.perceive(message=m1, max_tokens=1)
    print(r1)
    # r2 = b.perceive(message=m1, max_tokens=1)
    # print(r2)

    score1, score2 = scores[r1.content + "F"]

    for i in range(1, 10):
        m1 = Message(time=i, content="In round " + str(i) + " you chose Option " + r1.content + " and the other player chose Option " + "F" + ". Thus, you won " + str(score1) + " points and the other player won " + str(score2) + " points You are currently playing round " + str(i+1) + ". Q: Which Option do you choose, Option J or Option F?", role="user")

        # m2 = Message(time=i, content="In round " + str(i) + " you chose Option " + r2.content + " and the other player chose Option " + r1.content + ". Thus, you won " + str(score2) + " points and the other player won " + str(score1) + " points. You are currently playing round " + str(i+1) + ". Q: Which Option do you choose, Option J or Option F?", role="user")

        print(m1)

                
        r1 = a.perceive(message=m1, max_tokens=10)
        print(r1.content)

        ans1 = ""
        for c in r1.content:
            if c == "J" or c == "F":
                ans1 = c

        # print(m2)
        # r2 = b.perceive(message=m1, max_tokens=10)
        # print(r2)

        score1, score2 = scores[ans1 + "F"]


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


    # i = 5
    # while(i < 30):
    #     n = b.perceive(message=r1, max_tokens=256)
    #     print(n)
    #     m = Message(time=i, content=n.content, role="user")
    #     i += 1
    #     r1 = a.perceive(message=m, max_tokens=256)
    #     r1 = Message(time=i, content=r1.content, role="user")
    #     print(r1)
    

if __name__ == '__main__':
    test()
