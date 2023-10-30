import random
import sys
import os

sys.path.insert(0, f"{os.path.dirname(os.path.realpath(__file__))}/../")

from typing import List, Tuple
import pandas as pd
import agent
from llama import Message
import yaml

personas = [
    {"name": "Johannes", "info": "Male, 52 years old", "description": "A conservative Swiss lawyer. Concerned about transportation and urban development."},
    {"name": "Anna", "info": "Female, 42 years old", "description": "A single mum. She's a vegan and advocates for more cultural events in the city."},
    {"name": "Liam", "info": "Male, 35 years old", "description": "A tech entrepreneur. Interested in sustainable solutions and technological advancements for the city."},
    {"name": "Elena", "info": "Female, 29 years old", "description": "An environmentalist. Passionate about green initiatives and nature conservation."},
    {"name": "Matthias", "info": "Male, 48 years old", "description": "A professor. Enthusiastic about educational projects and intellectual events."},
    {"name": "Sophie", "info": "Female, 26 years old", "description": "An artist. Values creativity and wants the city to have more artistic ventures."},
    {"name": "Oscar", "info": "Male, 55 years old", "description": "A retired military officer. Prioritizes safety, security, and organized public services."},
    {"name": "Isabelle", "info": "Female, 31 years old", "description": "A chef. Wishes for more communal spaces and events promoting local cuisine and sustainable farming."},
    {"name": "Noah", "info": "Male, 28 years old", "description": "A fitness trainer. Advocates for health, wellness, and public recreational facilities."},
    {"name": "Eva", "info": "Female, 40 years old", "description": "A travel blogger. Hopes for projects that can enhance tourism and the city's global reputation."}
]


projects = '''
"Id,Name,District,Category,Cost
#1,Planting Workshops at Oerlikon 🌿,🟥 Nord,🌲 Nature,"5,000"
#2,Footpath Gardens around Train Stations 🌻,🟥 Nord,🌲 Nature,"10,000"
#3,Kid Festival at Leutschenpark 👶🏻,🟥 Nord,🎻 Culture,"5,000"
#4,Music Studio at Kulturbahnhof Affoltern 🎸,🟥 Nord,🎻 Culture,"10,000"
#5,Safe Bike Paths around Irchel Park 🚴🏾‍♀️,🟥 Nord,🚲 Transportation,"5,000"
#6,More Night Buses to Oerlikon 🚎,🟥 Nord,🚲 Transportation,"10,000"
#7,Free Open Badi Space in Wollishofen 🏊🏼‍♂️,🟨 Süd,🌲 Nature,"5,000"
#8,A Neighborhood Garden for Wiedikon 🏡,🟨 Süd,🌲 Nature,"10,000"
#9,Environmental Film Session for Kids 🎬,🟨 Süd,🎻 Culture,"5,000"
#10,Car-free Sunday Festivals on Mutschellenstrasse 🎉,🟨 Süd,🎻 Culture,"10,000"
#11,Free Bike Fixing Workshops 🚴🏼‍♀️,🟨 Süd,🚲 Transportation,"5,000"
#12,Car Sharing System for Young People 🚗,🟨 Süd,🚲 Transportation,"10,000"
#13,Transforming City Spaces under Trees into Gardens 🌵,🟦 Ost,🌲 Nature,"5,000"
#14,More Trees in Bellevue & Sechseläutenplatz 🌳,🟦 Ost,🌲 Nature,"10,000"
#15,Plant Festival in the City Centre 🪴,🟦 Ost,🎻 Culture,"5,000"
#16,Multicultural Festival at Sechseläutenplatz  🍻,🟦 Ost,🎻 Culture,"10,000"
#17,Bike Lanes on Seefeldstrasse 🚴🏽‍♂️,🟦 Ost,🚲 Transportation,"5,000"
#18,Speed bumps in the City and the Lake Front 🚙,🟦 Ost,🚲 Transportation,"10,000"
#19,Bird Houses for Zurich-Höngg 🦜,🟩 West,🌲 Nature,"5,000"
#20,Wet Biotope as Learning Garden in Frankental 👩🏼‍🏫,🟩 West,🌲 Nature,"10,000"
#21,Dingtheke: Community Things Exchange Library in Wipkingen 📚,🟩 West,🎻 Culture,"5,000"
#22,Sustainable Cooking Workshop with Kids 🧑🏼‍🍳,🟩 West,🎻 Culture,"10,000"
#23,Public Bicycle Moving Trailer to be Borrowed 📦,🟩 West,🚲 Transportation,"5,000"
#24,Car-free Langstrasse 🎉,🟩 West,🚲 Transportation,"10,000"
'''

header, *lines = projects.strip().split('\n')
random.shuffle(lines)
randomized_projects = '\n'.join([header] + lines)
vote_counts = {i: 0 for i in range(1, 25)}

def create_initial_context(persona):
    content = (f"Your name is {persona['name']}. {persona['description']}."
               " You're participating in a citywide Participatory Budgeting vote to decide on how a 50,000 budget should be spent in Zurich."
               " According to your personal preference, put your most prefered 5 projects in a list, using their project IDs with a hashtag in the front ." 
               " Make sure you read through all the projects and make the decision in accordance to your persona. Keep message very short, clear, direct, and concise.")
    return Message(time=0, content=content, role="system")

def run_pb_voting(n_steps: int = 10,
                  trigger_sentence: Message = None,
                  max_tokens: int = 600) -> List[Tuple[int, Message]]:
    agents = [agent.Agent(aid=i, recall=10, initial_context=create_initial_context(persona))
              for i, persona in enumerate(personas)]
    trigger_sentence = trigger_sentence if trigger_sentence is not None else Message(
        time=1,
        content=(
                "What city projects do you think should be funded? Select 5 projects using '#' and project id out of the following list:" + randomized_projects),
        role="user"
    )

    responses = []

    for i in range(min(n_steps, len(agents))):
        current_agent = agents[i]

        response = current_agent.perceive(message=trigger_sentence, max_tokens=max_tokens)

        selected_projects = [int(p[1:]) for p in response.content.split() if p.startswith("#") and p[1:].isdigit()]

        for p in selected_projects:
            if p in vote_counts:
                vote_counts[p] += 1

        responses.append((current_agent.id, response))

        print(responses[-1][0], responses[-1][1].content)

    print("\nVote Counts for Each Project:")
    sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
    for pid, count in sorted_votes:
        print(f'Project {pid}: {count} votes')

    return responses

if __name__ == '__main__':
    messages = run_pb_voting(n_steps=100, max_tokens=500)