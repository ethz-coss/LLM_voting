import pandas as pd
import os
import csv
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

import agent
from llama import Message

def calculate_accuracy_and_recall(agent_votes, real_votes):
    true_positives = len(set(agent_votes) & set(real_votes))
    accuracy = true_positives / len(agent_votes) if agent_votes else 0
    recall = true_positives / len(real_votes) if real_votes else 0
    return accuracy, recall

def create_initial_context(persona):
    description = persona.get('Description', '')
    # print(description)
    return Message(time=0, content=description, role="system")

def get_top_votes(vote_counts):
    return ', '.join(f"{proj_id}: {count}" for proj_id, count in sorted(vote_counts.items(), key=lambda item: item[1], reverse=True)[:5])

def calculate_average(stats, key):
    return sum(stat[key] for stat in stats) / len(stats) if stats else 0

def run_pb_voting(n_steps, max_tokens, projects, personas, source_file_name):
    agents = [agent.Agent(aid=i, recall=10, initial_context=create_initial_context(persona), temperature=0) for i, persona in enumerate(personas)]
    all_stats = []
    vote_counts_agent = defaultdict(int)
    vote_counts_real = defaultdict(int)
    vote_counts_random = defaultdict(int)
    projects_header = "#Id; Title; Cost; Location; Category\n"

    detailed_stats_path = f"outcome/{os.path.splitext(source_file_name)[0]}_detailed_stats.csv"
    with open(detailed_stats_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(personas[0].keys()) + ['response', 'agent_votes', 'random_votes',
                                                                             'real_votes', 'agent_accuracy',
                                                                             'agent_recall', 'random_accuracy',
                                                                             'random_recall'])
        writer.writeheader()

    for i in range(min(n_steps, len(agents))):
        current_agent = agents[i]
        persona = personas[i]
        real_votes = set(persona['vote'])
        n_proj = len(real_votes)

        random_projects = random.sample(projects, len(projects))
        projects_info = projects_header + '\n'.join(
            [f"#{project['Id']}; {project['Title']}; {project['Cost']}; {project['Location']}; {project['Category']}"
             for project in random_projects])

        trigger_content = (
            f"As part of a citywide participatory budgeting exercise, you have the opportunity to help decide how a budget of $50,000 should be allocated. "
            f"Think about your prefernce in location and category of urban projects. Vote as your assigned persona. Below is a list of potential projects for funding:"
            f"{projects_info}\n"
            f"Please select and respond with the IDs of up to {n_proj} project(s) you think should be funded after reading the whole list of projects. "
            f"List your chosen projects by their IDs, prefixed with '#', in a simple, comma-separated format."
        )

        trigger_sentence = Message(time=1, content=trigger_content, role="user")
        response = current_agent.perceive(message=trigger_sentence, max_tokens=max_tokens)
        agent_votes = set(int(match.group(1)) for match in re.finditer(r'#(\d+)', response.content))
        random_votes = set(random.sample([p['Id'] for p in projects], n_proj))

        agent_accuracy, agent_recall = calculate_accuracy_and_recall(agent_votes, real_votes)
        random_accuracy, random_recall = calculate_accuracy_and_recall(random_votes, real_votes)

        formatted_response = response.content.replace('\n', ' ')

        stats = persona.copy()
        stats.update({
            'agent_votes': str(agent_votes),
            'random_votes': str(random_votes),
            'real_votes': str(real_votes),
            'agent_accuracy': agent_accuracy,
            'agent_recall': agent_recall,
            'random_accuracy': random_accuracy,
            'random_recall': random_recall,
            'response': formatted_response
        })

        all_stats.append(stats)
        with open(detailed_stats_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=stats.keys())
            writer.writerow(stats)

        print(formatted_response)

        for vote in agent_votes:
            vote_counts_agent[vote] += 1
        for vote in real_votes:
            vote_counts_real[vote] += 1
        for vote in random_votes:
            vote_counts_random[vote] += 1

        updated_average_agent_accuracy = calculate_average(all_stats, 'agent_accuracy')
        updated_average_agent_recall = calculate_average(all_stats, 'agent_recall')
        updated_average_random_accuracy = calculate_average(all_stats, 'random_accuracy')
        updated_average_random_recall = calculate_average(all_stats, 'random_recall')

        print(
            f"A{current_agent.id} | "
            f"Acc/Rec: {updated_average_agent_accuracy:.1%}/{updated_average_agent_recall:.1%} | "
            f"Ran Acc/Rec: {updated_average_random_accuracy:.1%}/{updated_average_random_recall:.1%}"
        )

if __name__ == '__main__':
    source_file_path = '../aarau_data/aarau_pb_vote.csv'
    projects_file_path = '../aarau_data/aarau_projects.csv'

    votes_df = pd.read_csv(source_file_path)
    projects_df = pd.read_csv(projects_file_path)

    votes_df = votes_df[votes_df['votes'].notna() & (votes_df['votes'] != '')]
    votes_df['vote'] = votes_df['votes'].apply(lambda x: [int(vote.strip()) for vote in x.split(',')])

    projects = projects_df.to_dict(orient='records')
    personas = votes_df.to_dict(orient='records')

    source_file_name = os.path.basename(source_file_path)
    run_pb_voting(n_steps=500, max_tokens=800, projects=projects, personas=personas, source_file_name=source_file_name)
