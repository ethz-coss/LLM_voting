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
    return Message(time=0, content=description, role="system")

def get_top_votes(vote_counts):
    return ', '.join(f"{proj_id}: {count}" for proj_id, count in sorted(vote_counts.items(), key=lambda item: item[1], reverse=True)[:5])

def calculate_average(stats, key):
    return sum(stat[key] for stat in stats) / len(stats) if stats else 0

def extract_factors_from_response(response_content):
    matches = re.findall(r'\[(.*?)\]', response_content)
    return matches  # This will be a list of all matches

def get_next_file_number(directory, pattern):
    existing_files = os.listdir(directory)
    highest_number = 0
    for file_name in existing_files:
        match = re.search(pattern, file_name)
        if match:
            number = int(match.group(1))
            highest_number = max(highest_number, number)
    return highest_number + 1

def parse_votes(vote_str):
    votes = vote_str.strip("[]").split(',')
    return {int(vote.strip().strip("'")) for vote in votes if vote.strip()}

def run_pb_voting(n_steps, max_tokens, projects, personas, source_file_name, detailed_stats_path):
    agents = [agent.Agent(aid=i, recall=10, initial_context=create_initial_context(persona), temperature=0) for i, persona in enumerate(personas)]
    all_stats = []
    vote_counts_agent = defaultdict(int)
    vote_counts_real = defaultdict(int)
    vote_counts_random = defaultdict(int)
    projects_header = "#Id; Title; Cost; Location; Category\n"

    with open(detailed_stats_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(personas[0].keys()) + [
            'agent_votes', 'random_votes', 'agent_accuracy', 'agent_recall',
            'random_accuracy', 'random_recall', 'factor', 'response'
        ])
        writer.writeheader()

    for i in range(min(n_steps, len(agents))):
        current_agent = agents[i]
        persona = personas[i]
        real_votes = set(persona['real_votes'])
        n_proj = len(real_votes)

        random_projects = random.sample(projects, len(projects))
        projects_info = projects_header + '\n'.join(
            [f"#{project['Id']}; {project['Title']}; {project['Cost']}; {project['Location']}; {project['Category']}"
             for project in random_projects])

        trigger_content = (
            "You are participating in a citywide participatory budgeting exercise with a budget of $50,000. "
            "Your task is to help decide how this budget should be allocated based on your preferences. "
            "Consider your assigned persona and their preferences in location and category of urban projects. "
            "Below is a list of potential projects for funding:\n"
            f"{projects_info}\n"
            f"You are to select up to {n_proj} project(s) that you think should be funded. "
            "Please read the entire list before making a decision. "
            "Note: The size of the project ID number should not influence your choice. "
            "In your response, use square brackets '[]' to include one key factor that most influences your decision. "
            "Then, list your chosen projects by their ID numbers, prefixed with a hashtag '#', in a simple, comma-separated format. "
        )

        trigger_sentence = Message(time=1, content=trigger_content, role="user")
        response = current_agent.perceive(message=trigger_sentence, max_tokens=max_tokens)
        agent_votes = set(int(match.group(1)) for match in re.finditer(r'#(\d+)', response.content))
        random_votes = set(random.sample([p['Id'] for p in projects], n_proj))

        agent_accuracy, agent_recall = calculate_accuracy_and_recall(agent_votes, real_votes)
        random_accuracy, random_recall = calculate_accuracy_and_recall(random_votes, real_votes)

        formatted_response = response.content.replace('\n', ' ')
        factors = extract_factors_from_response(formatted_response)

        stats = persona.copy()
        stats.update({
            'agent_votes': str(agent_votes),
            'random_votes': str(random_votes),
            'agent_accuracy': agent_accuracy,
            'agent_recall': agent_recall,
            'random_accuracy': random_accuracy,
            'random_recall': random_recall,
            'factors': str(factors),
            'response': formatted_response
        })

        all_stats.append(stats)
        with open(detailed_stats_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=stats.keys())
            writer.writerow(stats)

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

        intersection_agent_real = real_votes & agent_votes
        intersection_random_real = real_votes & random_votes

        print(
            f"A{current_agent.id} | "
            f"Age: {persona['Age']} | "
            f"Gender: {persona['Gender']} | "
            f"Politics: {persona['Politics']} | "
            f"Description: {persona['Description']} | "
            f"Topics: {persona['Topics']} | "
            f"Key factor: {factors} | "
            f"Real Votes: {persona['real_votes']} | "
            f"Agent Votes: {agent_votes} | "
            f"Random Votes: {random_votes} | "
            f"Intersect AR: {intersection_agent_real} | "
            f"Intersect RR: {intersection_random_real} | "
            f"Acc/Rec: {updated_average_agent_accuracy:.1%}/{updated_average_agent_recall:.1%} | "
            f"Ran Acc/Rec: {updated_average_random_accuracy:.1%}/{updated_average_random_recall:.1%}"
        )

        print(formatted_response)

if __name__ == '__main__':
    model_name = '70bQ80'
    source_file_path = 'aarau_data/processed/aarau_pb_vote_description.csv'
    projects_file_path = 'aarau_data/processed/aarau_projects.csv'

    print(f"Source File: {source_file_path}")
    votes_df = pd.read_csv(source_file_path)
    projects_df = pd.read_csv(projects_file_path)

    print(f"Total Columns: {votes_df.shape[1]}")

    votes_df = votes_df[votes_df['Votes'].notna() & (votes_df['Votes'] != '')]

    votes_df['real_votes'] = votes_df['Votes'].apply(parse_votes)
    votes_df = votes_df.drop(columns=['Votes'])

    projects = projects_df.to_dict(orient='records')
    personas = votes_df.to_dict(orient='records')

    target_directory = 'aarau_outcome/agent_vote'
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    new_file_number = get_next_file_number(target_directory, rf'aarau_pb_vote_{model_name}_(\d+).csv')
    new_file_name = f'aarau_pb_vote_{model_name}_{new_file_number}.csv'
    detailed_stats_path = os.path.join(target_directory, new_file_name)

    run_pb_voting(n_steps=500, max_tokens=800, projects=projects, personas=personas,
                  source_file_name=source_file_path, detailed_stats_path=detailed_stats_path)
