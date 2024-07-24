import json
from typing import List
import re
import sys
import csv
import os

sys.path.insert(0, f"{os.path.dirname(os.path.realpath(__file__))}/../")
import agent
from llama import Message

num_voter = 180

def get_next_file_number(directory, pattern):
    max_number = 0
    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            number = int(match.group(1))
            max_number = max(max_number, number)
    return max_number + 1

def load_descriptions(csv_file_path):
    descriptions = []
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            descriptions.append(row['Description'])
    return descriptions

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
appr_ins = "Select any number of projects."
kapp_ins = "Select exactly 5 projects."
cumu_ins = "Distribute 10 points among the projects you like. List each of the project id you choose and the points you allocate together."
rank_ins = "Select 5 projects and rank them from the most preferred to the 5th most preferred."

instruction_labels = {
    appr_ins: "Approval Voting",
    kapp_ins: "5-Approval Voting",
    cumu_ins: "Cumulative Voting",
    rank_ins: "Rank Voting",
    "reversed_order": "5-Approval Voting (Reversed Order)",
    "reversed_id": "5-Approval Voting (Reversed IDs)"
}

instructions = [appr_ins, kapp_ins, cumu_ins, rank_ins]

header, *lines = projects.strip().split('\n')

def generate_file_paths(target_directory, model_name, label, suffix, is_json=False):
    file_extension = 'json' if is_json else 'csv'
    pattern = rf'{suffix}_{model_name}_{label}_(\d+).{file_extension}'
    file_number = get_next_file_number(target_directory, pattern)
    file_name = f'{suffix}_{model_name}_{label}_{file_number}.{file_extension}'
    return os.path.join(target_directory, file_name)


def get_project_info(projects):
    project_info = {}
    lines = projects.strip().split('\n')[1:]
    for line in lines:
        parts = line.split(',')
        project_id = int(parts[0][1:])
        project_info[project_id] = parts[1:]
    return project_info

project_info = get_project_info(projects)

def save_outcome_to_csv(sorted_votes, project_info, file_path):
    with open(file_path, mode='w', newline='') as csv_file:
        fieldnames = ['Rank', 'Votes', 'Id', 'Name', 'District', 'Category', 'Cost']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, (project_id, total_votes) in enumerate(sorted_votes, start=1):
            project_data = project_info[project_id]
            row = {
                'Rank': rank,
                'Votes': total_votes,
                'Id': f"#{project_id}",
                'Name': project_data[0],
                'District': project_data[1],
                'Category': project_data[2],
                'Cost': project_data[3]
            }
            writer.writerow(row)

def reverse_project_list(projects):
    lines = projects.strip().split('\n')
    header = lines[0]
    reversed_lines = lines[:0:-1]
    reversed_projects = '\n'.join([header] + reversed_lines)
    return reversed_projects

def reverse_project_ids(projects):
    lines = projects.strip().split('\n')
    header = lines[0]
    project_lines = lines[1:]
    num_projects = len(project_lines)
    reversed_id_projects = []

    for line in project_lines:
        parts = line.split(',')
        original_id = int(parts[0][1:])
        new_id = f"#{num_projects - original_id + 1}"
        reversed_id_projects.append(new_id + "," + ",".join(parts[1:]))

    reversed_projects = '\n'.join([header] + reversed_id_projects)
    return reversed_projects

def parse_cumu_votes(response):
    point_allocations = re.findall(r'(#\d+(?:, #\d+)*)(.*?)\s*(\d+)\s*point', response, re.IGNORECASE)
    total_points = sum(int(points) for _, _, points in point_allocations)
    votes = {}
    for project_group, _, points in point_allocations:
        project_ids = re.findall(r'#(\d+)', project_group)
        normalized_points = (int(points) / total_points) * 10 if total_points != 0 else 0
        for project_id in project_ids:
            votes[int(project_id)] = normalized_points / len(project_ids)

    return votes


def parse_rank_votes(response):
    project_ids = re.findall(r'#(\d+)', response)
    ranks = {int(project_ids[i]): i + 1 for i in range(min(5, len(project_ids)))}
    if any(rank > 5 for rank in ranks.values()):
        print(f"INVALID RANKS: {ranks}")
        print(project_ids)

    return ranks


def borda_score(ranks):
    points = {project: 5 - rank + 1 for project, rank in ranks.items()}
    print(ranks)
    return points

# # Old experiments without persona description
# def create_initial_context(description):
#     content = (
#         f"In this participatory budgeting program, you will be voting for the allocation of the 60,000 CHF on city projects in Zurich. You are a university student in Zurich. Think about what projects you would personally like to be funded here. Make sure your votes reflects your preference. The explanation should be very short."
#     )
#     return Message(time=0, content=content, role="system")

# Updated experiments include persona description
def create_initial_context(description):
    content = (
        f"In this participatory budgeting program, you will be voting for the allocation of the 50,000 CHF on city projects in Zurich. {description} Think about what projects you would personally like to be funded here. Make sure your votes reflect your preference. The explanation should be very short."
    )
    return Message(time=0, content=content, role="system")


def run_pb_voting(instruction, descriptions, reversed=False, id_reversed=False, n_steps: int = 180, max_tokens: int = 600, temp: float = 1) -> List[dict]:
    agents = [agent.Agent(aid=i, recall=2, initial_context=create_initial_context(descriptions[i]), temperature=temp) for i in range(num_voter)]
    vote_counts = {i: 0 for i in range(1, 25)}
    voting_data = []

    print("\n======LOADING=DEMOCRACY=======")
    instruction_label = instruction_labels.get(instruction)
    print("\nVoting Method: " + instruction_label + "\n")

    if reversed:
        project_display = reverse_project_list(projects)
        print("REVERSED ORDER LIST:")
        print(project_display+"\n")
    elif id_reversed:
        print("REVERSED ID LIST:")
        project_display = reverse_project_ids(projects)
        print(project_display+"\n")
    else:
        project_display = projects

    for i in range(min(n_steps, len(agents))):
        current_agent = agents[i]
        initial_context_content = current_agent.initial_context.content

        trigger_sentence = Message(
            time=1,
            content=(
                "Look at 24 projects in the following table and think about your preferences in urban projects before you start voting."
                f"{project_display}"
                "Select projects out of your personal interest using '#' and project id."
                f"{instruction}"
            ),
            role="user"
        )
        trigger_sentence_content = trigger_sentence.content

        response = current_agent.perceive(
            message=trigger_sentence, max_tokens=max_tokens)
        formatted_response = response.content.replace('\n', ' ')

        if instruction == cumu_ins:
            votes = parse_cumu_votes(formatted_response)
            for project_id, points in votes.items():
                vote_counts[project_id] += points
        elif instruction == rank_ins:
            ranks = parse_rank_votes(formatted_response)
            votes = borda_score(ranks)
            for project_id, points in votes.items():
                vote_counts[project_id] += points
        else:
            selected_projects = {int(match.group(1)) for match in re.finditer(r'#(\d+)', formatted_response)}
            for p in selected_projects:
                if p in vote_counts:
                    vote_counts[p] += 1
            votes = list(selected_projects)

        if id_reversed:
            votes = [25 - p for p in votes]

        voting_data.append({
            'agent_id': current_agent.id,
            'votes': votes,
            'temperature': temp,
            'response': formatted_response,
            'initial_context': initial_context_content,
            'trigger_sentence': trigger_sentence_content
        })

        print(f"AGENT {current_agent.id}")
        print(f"Votes: {votes}")
        print(f"Response: {formatted_response}\n")
        print_top_votes(vote_counts)
        print(" ")

    outcome = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
    return voting_data, outcome

def pb_voting_cot(instruction, descriptions, reversed=False, id_reversed=False, n_steps: int = 180, max_tokens: int = 600, temp: float = 1) -> List[dict]:
    agents = [agent.Agent(aid=i, recall=2, initial_context=create_initial_context(descriptions[i]), temperature=temp) for i in range(num_voter)]
    vote_counts = {i: 0 for i in range(1, 25)}
    voting_data = []
    # print("\n======LOADING=DEMOCRACY=======")
    instruction_label = instruction_labels.get(instruction)
    # print("\nVoting Method: " + instruction_label + "\n")

    if reversed:
        project_display = reverse_project_list(projects)
        print("REVERSED ORDER LIST:")
        print(project_display+"\n")
    elif id_reversed:
        print("REVERSED ID LIST:")
        project_display = reverse_project_ids(projects)
        print(project_display+"\n")
    else:
        project_display = projects

    for i in range(min(n_steps, len(agents))):
        current_agent = agents[i]
        initial_context_content = current_agent.initial_context.content

        # First prompt to generate the thought process
        thought_trigger = Message(
            time=1,
            content=(
                "Consider your personal interests and explain how they might influence your decisions when choosing urban projects in a participatory budgeting program in Zurich."
            ),
            role="user"
        )
        thought_response = current_agent.perceive(
            message=thought_trigger, max_tokens=max_tokens).content

        # Second prompt to vote based on the thought process
        vote_trigger = Message(
            time=2,
            content=(
                "Look at 24 projects in the following table and think about your preferences in urban projects before you start voting."
                f"{project_display}"
                "Select projects out of your personal interest using '#' and project id."
                f"{instruction}"
            ),
            role="user"
        )
        vote_response = current_agent.perceive(
            message=vote_trigger, max_tokens=max_tokens).content.replace('\n', ' ')

        if instruction == cumu_ins:
            votes = parse_cumu_votes(vote_response)
            for project_id, points in votes.items():
                vote_counts[project_id] += points
        elif instruction == rank_ins:
            ranks = parse_rank_votes(vote_response)
            votes = borda_score(ranks)
            for project_id, points in votes.items():
                vote_counts[project_id] += points
        else:
            selected_projects = {int(match.group(1)) for match in re.finditer(r'#(\d+)', vote_response)}
            for p in selected_projects:
                if p in vote_counts:
                    vote_counts[p] += 1
            votes = list(selected_projects)

        if id_reversed:
            votes = [25 - p for p in votes]

        voting_data.append({
            'agent_id': current_agent.id,
            'votes': votes,
            'temperature': temp,
            'thought_response': thought_response,
            'vote_response': vote_response,
            'initial_context': initial_context_content,
            'trigger_thought': thought_trigger.content,
            'trigger_vote': vote_trigger.content
        })
        print('==============================')
        print(f"Agent ID: {current_agent.id}\n")
        print(f"Thoughts:\n {thought_response}\n")
        print(f"Voting Response: {vote_response}\n")
        print(f"Votes: {votes}\n")
        print_top_votes(vote_counts)
        print(" ")

    outcome = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
    return voting_data, outcome


def print_top_votes(vote_counts, top_n=10):
    sorted_votes = sorted(vote_counts.items(),
                          key=lambda x: x[1], reverse=True)
    top_votes_str = ", ".join(
        [f'#{pid}: {count}' for pid, count in sorted_votes[:top_n]])
    print(f"Top {top_n} Voted Projects: {top_votes_str}")


def save_results_to_csv(voting_data, file_path):
    with open(file_path, mode='w', newline='') as csv_file:
        fieldnames = ['agent_id', 'votes', 'response', 'initial_context', 'trigger_sentence']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for data in voting_data:
            writer.writerow(data)


def save_results_to_json(voting_data, file_path):
    with open(file_path, mode='w') as json_file:
        json.dump(voting_data, json_file, indent=4)


if __name__ == '__main__':
    model_name = "llama2_cot"
    # temperature_settings = [0, 0.5, 1, 1.5, 2] 
    temperature_settings = [1] # default temperature
    descriptions = load_descriptions('data/lab_meta.csv')  

    for temp in temperature_settings:
        temp_str = str(temp).replace('.', 'p')  
        target_directory = f'outcome/{model_name}_vote_temp{temp_str}'
        
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # all_instructions = instructions + [kapp_ins, kapp_ins]
        # labels = ['appr', 'kapp', 'cumu', 'rank', 'reversed_order', 'reversed_id']
            
        instructions = [kapp_ins]  
        labels = ['kapp']  # default voting method

        for ins_index, instruction in enumerate(instructions):
            # reversed = ins_index == 0  # Set these flags for reversed order experiment
            # id_reversed = ins_index == 1  # Set these flags for reversed order id experiment
            label = labels[ins_index]
            
            
            # votes, outcome = run_pb_voting(instruction, reversed=False, id_reversed=False, n_steps=5, max_tokens=600, temp = temp, descriptions=descriptions)
            votes, outcome = pb_voting_cot(instruction, reversed=False, id_reversed=False, max_tokens=600, temp = temp, descriptions=descriptions) # For CoT experiments

            vote_path = generate_file_paths(target_directory, model_name, label, 'votes', is_json=True)
            outcome_path = generate_file_paths(target_directory, model_name, label, 'outcome')

            save_results_to_json(votes, vote_path)
            save_outcome_to_csv(outcome, project_info, outcome_path)

            print(f"\nCompleted: {instruction_labels[instruction if ins_index < 4 else label]} Voting")