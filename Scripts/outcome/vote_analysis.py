import pandas as pd
import ast
import os

input_file = 'aarau_pb_vote_sm_2.csv'
base_name = os.path.splitext(input_file)[0]
output_file = f'{base_name}_insights.json'

df = pd.read_csv(input_file)

insights_df = pd.DataFrame()

insights_df['Average Age'] = [df['Age'].mean()]
insights_df['Gender Ratio (Women:Men)'] = [df['Gender'].value_counts().get('woman', 0) / df['Gender'].value_counts().get('man', 1)]

insights_df['Percentage of non-Schweiz Nationality'] = [(df['Nationalitaet'] != 'Schweiz').mean() * 100]

politics_counts = df['Politics'].value_counts()
top_preferences = df['Top Preferences'].str.get_dummies(sep=', ').sum()

insights_df = insights_df.assign(**politics_counts, **top_preferences)

def parse_votes(vote_str):
    if vote_str.startswith("{"):
        vote_str = vote_str.replace("{", "[").replace("}", "]")
    return ast.literal_eval(vote_str)

vote_categories = ['agent_votes', 'random_votes', 'real_votes']
vote_counts = {category: {} for category in vote_categories}

for category in vote_categories:
    for vote_str in df[category]:
        for vote in parse_votes(vote_str):
            vote_key = f"#{vote}"
            vote_counts[category][vote_key] = vote_counts[category].get(vote_key, 0) + 1

for category, counts in vote_counts.items():
    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    insights_df[category] = [sorted_counts]

insights_df['Average Agent Accuracy'] = [df['agent_accuracy'].mean()]
insights_df['Average Agent Recall'] = [df['agent_recall'].mean()]
insights_df['Average Random Accuracy'] = [df['random_accuracy'].mean()]
insights_df['Average Random Recall'] = [df['random_recall'].mean()]

insights_df.to_json(output_file, orient='records', lines=True)
