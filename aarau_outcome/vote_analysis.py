import pandas as pd
import ast
import os
import json
import sys

if len(sys.argv) != 2:
    print("Usage: python vote_analysis.py <input_file_name>")
    sys.exit(1)

input_file_name = sys.argv[1]
input_file_path = os.path.join('aarau_outcome/agent_vote', input_file_name)  # Adjust path according to the structure

base_name = os.path.basename(os.path.splitext(input_file_path)[0])
output_file = f'vote_analysis/{base_name}_insights.json'

if not os.path.exists(input_file_path):
    print(f"The file {input_file_path} does not exist.")
    sys.exit(1)

output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df = pd.read_csv(input_file_path)

def get_age_group(age):
    if 0 <= age <= 29:
        return '0-29'
    elif 30 <= age <= 44:
        return '30-44'
    elif 45 <= age <= 64:
        return '45-64'
    else:
        return '65+'

def categorize_politics(political_stance):
    if political_stance in ['very left', 'left-leaning']:
        return 'Left'
    elif political_stance in ['very right', 'right-leaning']:
        return 'Right'
    else:
        return 'Centre'

def is_highly_educated(education_level):
    highly_educated_levels = ['highly educated']  # Define the levels that count as 'Highly Educated'
    return education_level in highly_educated_levels

def has_preference(preference_list, preference):
    if isinstance(preference_list, str):
        return preference in preference_list.split(', ')
    return False

def categorize_children_info(children_info):
    if children_info == 'Has children.':
        return 'Has Children'
    else:
        return 'No Children or Unknown'


df['Children Category'] = df['Children Info'].apply(categorize_children_info)

df['Age Group'] = df['Age'].apply(get_age_group)
unique_preferences = set()
df['Top Preferences'].dropna().str.split(', ').apply(unique_preferences.update)


df['Top Preference'] = df['Top Preferences'].apply(lambda x: x if isinstance(x, str) else '')
df['Political Group'] = df['Politics'].apply(categorize_politics)

# Apply the function to create a new column
df['Education Group'] = df['Education'].apply(lambda x: 'Highly Educated' if is_highly_educated(x) else 'Not Highly Educated')



insights_df = pd.DataFrame()

insights_df['Average Age'] = [df['Age'].mean()]
insights_df['Gender Ratio (Women:Men)'] = [df['Gender'].value_counts().get('woman', 0) / df['Gender'].value_counts().get('man', 1)]

insights_df['Percentage of non-Schweiz Nationality'] = [(df['Nationalitaet'] != 'Schweiz').mean() * 100]

politics_counts = df['Politics'].value_counts()
top_preferences = df['Top Preferences'].str.get_dummies(sep=', ').sum()

insights_df = insights_df.assign(**politics_counts, **top_preferences)

def parse_votes(vote_str):
    try:
        if isinstance(vote_str, str) and vote_str.startswith("{"):
            vote_str = vote_str.replace("{", "[").replace("}", "]")
        return ast.literal_eval(vote_str) if isinstance(vote_str, str) else []
    except (ValueError, SyntaxError):
        print(vote_str)
        return []

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

def aggregate_votes_by_group(df, group_column, vote_column):
    group_vote_counts = {}
    for _, row in df.iterrows():
        group_value = row[group_column]
        votes = parse_votes(row[vote_column])
        for vote in votes:
            vote_key = f"#{vote}"
            group_vote_counts.setdefault(group_value, {}).setdefault(vote_key, 0)
            group_vote_counts[group_value][vote_key] += 1
    return group_vote_counts

demographic_columns = ['Age Group', 'Gender', 'Political Group', 'Education Group', 'Birthplace Info', 'Children Category']
grouped_votes = {group: {'Real Votes': aggregate_votes_by_group(df, group, 'real_votes'), 'Agent Vote': aggregate_votes_by_group(df, group, 'agent_votes')} for group in demographic_columns}

combined_insights = {
    'General Insights': insights_df.to_dict(orient='records'),
    'Demographic Insights': grouped_votes
}

def print_top_votes(grouped_votes, top_n=5):
    for demographic, votes in grouped_votes.items():
        print(f"Demographic: {demographic}")
        for vote_type, vote_data in votes.items():
            print(f"  {vote_type}:")
            for group, group_votes in vote_data.items():
                sorted_votes = dict(sorted(group_votes.items(), key=lambda item: item[1], reverse=True))
                top_votes = list(sorted_votes.items())[:top_n]
                top_votes_str = ', '.join([f"{project}: {count}" for project, count in top_votes])
                print(f"    {group}: {top_votes_str}")
            print()
        print()

print_top_votes(grouped_votes)

def aggregate_votes_by_specific_preference(df, preference, vote_column):
    preference_vote_counts = {}
    for _, row in df.iterrows():
        if has_preference(row['Top Preferences'], preference):
            votes = parse_votes(row[vote_column])
            for vote in votes:
                vote_key = f"#{vote}"
                preference_vote_counts[vote_key] = preference_vote_counts.get(vote_key, 0) + 1
    return preference_vote_counts

preference_vote_results = {}
for preference in unique_preferences:
    real_votes = aggregate_votes_by_specific_preference(df, preference, 'real_votes')
    agent_votes = aggregate_votes_by_specific_preference(df, preference, 'agent_votes')
    preference_vote_results[preference] = {'Real Votes': real_votes, 'Agent Votes': agent_votes}

for preference, votes in preference_vote_results.items():
    print(f"Preference: {preference}")
    for vote_type, vote_counts in votes.items():
        print(f"  {vote_type}:")
        sorted_votes = dict(sorted(vote_counts.items(), key=lambda item: item[1], reverse=True))
        top_votes = list(sorted_votes.items())[:5]
        print(f"    Top Votes: {', '.join([f'{project}: {count}' for project, count in top_votes])}")
    print()

output_file_name = os.path.splitext(input_file_name)[0] + '_insights.json'
output_file_path = os.path.join('vote_analysis', output_file_name)

with open(output_file_path, 'w') as file:
    json.dump(combined_insights, file, indent=4)
print(f"Insights saved to {output_file_path}")