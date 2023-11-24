import random
import re
import sys
import csv
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
Id,Name,District,Category,Cost
#1,Planting Workshops at Oerlikon,Nord,Nature,5000
#2,Footpath Gardens around Train Stations,Nord,Nature,10000
#3,Kid Festival at Leutschenpark,Nord,Culture,5000
#4,Music Studio at Kulturbahnhof Affoltern,Nord,Culture,10000
#5,Safe Bike Paths around Irchel Park,Nord,Transportation,5000
#6,More Night Buses to Oerlikon,Nord,Transportation,10000
#7,Free Open Badi Space in Wollishofen,Süd,Nature,5000
#8,A Neighborhood Garden for Wiedikon,Süd,Nature,10000
#9,Environmental Film Session for Kids,Süd,Culture,5000
#10,Car-free Sunday Festivals on Mutschellenstrasse,Süd,Culture,10000
#11,Free Bike Fixing Workshops,Süd,Transportation,5000
#12,Car Sharing System for Young People,Süd,Transportation,10000
#13,Transforming City Spaces under Trees into Gardens,Ost,Nature,5000
#14,More Trees in Bellevue & Sechseläutenplatz,Ost,Nature,10000
#15,Plant Festival in the City Centre,Ost,Culture,5000
#16,Multicultural Festival at Sechseläutenplatz,Ost,Culture,10000
#17,Bike Lanes on Seefeldstrasse,Ost,Transportation,5000
#18,Speed bumps in the City and the Lake Front,Ost,Transportation,10000
#19,Bird Houses for Zurich-Höngg,West,Nature,5000
#20,Wet Biotope as Learning Garden in Frankental,West,Nature,10000
#21,Dingtheke: Community Things Exchange Library in Wipkingen,West,Culture,5000
#22,Sustainable Cooking Workshop with Kids,West,Culture,10000
#23,Public Bicycle Moving Trailer to be Borrowed,West,Transportation,5000
#24,Car-free Langstrasse,West,Transportation,10000
'''

data = pd.read_csv("../lab_data/lab_meta.csv")

personas = [{"ID": row["ID"],
             "name": f"Participant {row['ID']}",
             "description": row["Description"]} for _, row in data.iterrows()]

header, *lines = projects.strip().split('\n')
random.shuffle(lines)
randomized_projects = '\n'.join([header] + lines)
vote_counts = {i: 0 for i in range(1, 25)}
test_value = 180

def create_initial_context(persona):
    content = (f"{persona['description']}."
               " You're participating in a citywide Participatory Budgeting vote to decide on how a 50,000 budget should be spent in Zurich."
               " According to your personal preference, put your most prefered 5 projects in a list, using their project IDs with a hashtag in the front ." 
               " Make sure you read through all the projects and make the decision in accordance to your persona. Keep message very short, clear, direct, and concise.")
    return Message(time=0, content=content, role="system")

def run_pb_voting(n_steps: int = 180,
                  trigger_sentence: Message = None,
                  max_tokens: int = 600) -> List[Tuple[int, Message]]:
    agents = [agent.Agent(aid=i, recall=10, initial_context=create_initial_context(persona), temperature=0)
              for i, persona in enumerate(personas)]
    trigger_sentence = trigger_sentence if trigger_sentence is not None else Message(
        time=1,
        content=(
                "What city projects do you think should be funded? Simply select 5 projects out of your personal interest using '#' and project id out of the following list:" + randomized_projects),
        role="user"
    )

    responses = []

    for i in range(min(n_steps, test_value)):
        current_agent = agents[i]

        response = current_agent.perceive(message=trigger_sentence, max_tokens=max_tokens)

        selected_projects = [int(match.group(1)) for match in re.finditer(r'#(\d+)', response.content)]
        print(selected_projects)

        for p in selected_projects:
            if p in vote_counts:
                vote_counts[p] += 1

        responses.append((current_agent.id, response))

        print(responses[-1][0], responses[-1][1].content)

    print("\nVote Counts for Each Project:")
    sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
    for pid, count in sorted_votes:
        print(f'Project {pid}: {count} votes')
    save_results_to_csv(sorted_votes)

    return responses


def save_results_to_csv(sorted_votes, projects):
    projects_data = [line.strip().split(',') for line in projects.strip().split('\n')]

    headers = projects_data[0] + ['Vote Count']
    projects_data = projects_data[1:]

    outcome_folder = 'aarau_outcome'
    if not os.path.exists(outcome_folder):
        os.makedirs(outcome_folder)

    csv_file_path = os.path.join(outcome_folder, 'lab_voting_results.csv')

    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)

        writer.writeheader()

        for pid, count in sorted_votes:
            project_line = projects_data[pid - 1]
            project_data = {
                'Id': project_line[0],
                'Name': project_line[1],
                'District': project_line[2],
                'Category': project_line[3],
                'Cost': project_line[4],
                'Vote Count': count
            }
            writer.writerow(project_data)

    return csv_file_path

if __name__ == '__main__':
    messages = run_pb_voting(n_steps=100, max_tokens=500)
