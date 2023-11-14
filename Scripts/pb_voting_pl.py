import pandas as pd
import os
import csv
import sys
import random
import re
from collections import defaultdict
from typing import List, Tuple
sys.path.insert(0, f"{os.path.dirname(os.path.realpath(__file__))}/../")

import agent
from llama import Message

def read_pb_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    meta_data = []
    projects_data = []
    votes_data = []
    current_section = None

    for line in lines:
        if 'META' in line:
            current_section = meta_data
        elif 'PROJECTS' in line:
            current_section = projects_data
        elif 'VOTES' in line:
            current_section = votes_data
        elif current_section is not None:
            if line.strip():
                current_section.append(line.strip().split(';'))

    meta_df = pd.DataFrame(meta_data[1:], columns=meta_data[0])
    projects_df = pd.DataFrame(projects_data[1:], columns=projects_data[0])
    votes_df = pd.DataFrame(votes_data[1:], columns=votes_data[0])

    projects_df = projects_df.astype({'project_id': int, 'cost': float, 'votes': int, 'selected': int})
    votes_df = votes_df.astype({'voter_id': int, 'age': int, 'vote': str})

    votes_df['vote'] = votes_df['vote'].apply(lambda x: [int(vote) for vote in x.split(',')])
    votes_df = votes_df[votes_df['vote'].apply(len) == 5]

    return meta_df, projects_df, votes_df

def calculate_accuracy_and_recall(agent_votes, real_votes):
    true_positives = len(set(agent_votes) & set(real_votes))
    predicted_positives = len(agent_votes)
    actual_positives = len(real_votes)

    accuracy = true_positives / predicted_positives if predicted_positives else 0
    recall = true_positives / actual_positives if actual_positives else 0
    return accuracy, recall

def create_initial_context(persona):
    age = persona['age']
    gender = 'woman' if persona['sex'] == 'F' else 'man'

    content = (f"You are a {age}-year-old {gender} living in Warsaw."
               "You can pick 5 projects in the participatory budgeting of your city. Here is a list of urban projects you can vote for. List out 5 projects you would select according to your persona. Make sure you read all the projects first.  "
               "List 5 projects of your choice using their IDs with a '#' in front. Your votes should reflect your personal perspective.")

    return Message(time=0, content=content, role="system")

def get_top_votes(vote_counts):
    top_votes = sorted(vote_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    return ', '.join([f"{proj_id}: {count}" for proj_id, count in top_votes])

def calculate_average(stats, key):
    return sum(stat[key] for stat in stats) / len(stats) if stats else 0


def run_pb_voting(n_steps, max_tokens, projects, personas, source_file_name):
    agents = [agent.Agent(aid=i, recall=10, initial_context=create_initial_context(personas[i]), temperature=0) for i in
              range(len(personas))]
    all_stats = []
    vote_counts_agent = defaultdict(int)
    vote_counts_real = defaultdict(int)
    vote_counts_random = defaultdict(int)

    file_base_name = source_file_name.split('/')[-1].split('.')[0]
    detailed_stats_path = f"outcome/{file_base_name}_detailed_stats.csv"

    counter = 1
    while os.path.exists(detailed_stats_path):
        detailed_stats_path = f"outcome/{file_base_name}_detailed_stats_{counter}.csv"
        counter += 1

    detailed_headers = ['agent_id', 'voter_id', 'age', 'gender', 'agent_votes', 'random_votes', 'real_votes',
                        'agent_accuracy', 'agent_recall', 'random_accuracy', 'random_recall']
    with open(detailed_stats_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=detailed_headers)
        writer.writeheader()

    for i in range(min(n_steps, len(agents))):
        current_agent = agents[i]
        persona = personas[i]
        voter_id = persona['voter_id']
        real_votes = persona['vote']
        age = persona['age']
        gender = persona['sex']

        random_projects = random.sample(projects, len(projects))
        projects_info = '\n'.join([f"{project['project_id']};{project['name']}" for project in random_projects])

        trigger_content = (
            f"What city projects do you think should be funded? "
            f"Think about your diverse interests as a {age}-year-old {gender} living in Warsaw. Keep it short. Read all projects first. "
            "Simply select 5 project(s) out of your personal interest using '#' and project id out of the following list:"
            f"{projects_info}"
        )
        trigger_sentence = Message(time=1, content=trigger_content, role="user")

        response = current_agent.perceive(message=trigger_sentence, max_tokens=max_tokens)
        print(response)
        agent_votes = [int(match.group(1)) for match in re.finditer(r'#(\d+)', response.content)]
        random_votes = random.sample([p['project_id'] for p in projects], 5)

        agent_accuracy, agent_recall = calculate_accuracy_and_recall(agent_votes, real_votes)
        random_accuracy, random_recall = calculate_accuracy_and_recall(random_votes, real_votes)

        for vote in agent_votes:
            if vote in vote_counts_agent:
                vote_counts_agent[vote] += 1
            else:
                vote_counts_agent[vote] = 1
        for vote in real_votes:
            vote_counts_real[vote] += 1
        for vote in random_votes:
            vote_counts_random[vote] += 1

        updated_average_agent_accuracy = calculate_average(all_stats, 'agent_accuracy')
        updated_average_agent_recall = calculate_average(all_stats, 'agent_recall')
        updated_average_random_accuracy = calculate_average(all_stats, 'random_accuracy')
        updated_average_random_recall = calculate_average(all_stats, 'random_recall')

        stats = {
            'agent_id': current_agent.id,
            'voter_id': voter_id,
            'age': age,
            'gender': gender,
            'agent_votes': agent_votes,
            'random_votes': random_votes,
            'real_votes': real_votes,
            'agent_accuracy': agent_accuracy,
            'agent_recall': agent_recall,
            'random_accuracy': random_accuracy,
            'random_recall': random_recall
        }
        all_stats.append(stats)

        with open(detailed_stats_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=detailed_headers)
            writer.writerow(stats)

        top_agent_votes = get_top_votes(vote_counts_agent)
        top_real_votes = get_top_votes(vote_counts_real)
        top_random_votes = get_top_votes(vote_counts_random)

        print(
            f"A{current_agent.id} | "
            f"AV: {top_agent_votes} | "
            f"RV: {top_real_votes} | "
            f"RanV: {top_random_votes} | "
            f"Acc/Rec: {updated_average_agent_accuracy:.1%}/{updated_average_agent_recall:.1%} | "
            f"Ran Acc/Rec: {updated_average_random_accuracy:.1%}/{updated_average_random_recall:.1%}"
        )


def create_outcome_table(projects_df, vote_counts):
    projects_df['Agent Votes'] = projects_df['project_id'].apply(lambda x: vote_counts.get(x, 0))
    projects_df['Difference'] = projects_df['Agent Votes'] - projects_df['votes']
    return projects_df[['project_id', 'name', 'votes', 'Agent Votes', 'Difference']]


def save_results_to_csv(sorted_votes, projects):
    headers = ['Id', 'Name', 'District', 'Category', 'Cost', 'Vote Count']
    outcome_folder = 'outcome'
    if not os.path.exists(outcome_folder):
        os.makedirs(outcome_folder)

    csv_file_path = os.path.join(outcome_folder, 'pl_voting_results.csv')

    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()

        for pid, count in sorted_votes:
            project_data = next((project for project in projects if project['project_id'] == pid), None)
            if project_data:
                project_data['Vote Count'] = count
                writer.writerow({key: project_data[key] for key in headers})

    return csv_file_path


if __name__ == '__main__':
    source_file_path = '../pb_projects/poland_warszawa_2018_bialoleka.pb'
    meta_df, projects_df, votes_df = read_pb_file(source_file_path)
    votes_df = votes_df[votes_df['vote'].apply(len) == 5]

    projects = projects_df.to_dict(orient='records')
    random.shuffle(projects)
    projects_info = '\n'.join([f"{project['project_id']};{project['name']}" for project in projects])

    personas = votes_df.to_dict(orient='records')
    vote_counts = {project['project_id']: 0 for project in projects}

    source_file_name = os.path.basename(source_file_path)
    messages = run_pb_voting(n_steps=500, max_tokens=500, projects=projects, personas=personas,
                             source_file_name=source_file_name)
