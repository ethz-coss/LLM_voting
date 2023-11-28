import sys
import subprocess
import os
from pathlib import Path


def get_next_file_number(directory, prefix, suffix):
    existing_files = Path(directory).glob(f"{prefix}_*{suffix}")
    max_number = 0
    for f in existing_files:
        parts = f.stem.split('_')
        try:
            number = int(parts[-1])
            max_number = max(max_number, number)
        except ValueError:
            continue
    return max_number



def run_voting_and_analysis(repetitions, model_name):
    for _ in range(repetitions):
        subprocess.run(['python', 'scripts/pb_voting_ar.py', model_name], check=True)

        target_directory = 'aarau_outcome/agent_vote/'
        new_file_number = get_next_file_number(target_directory, f'aarau_pb_vote_{model_name}', '.csv')
        new_file_name = f'aarau_pb_vote_{model_name}_{new_file_number}.csv'
        print(new_file_name)
        subprocess.run(['python', 'aarau_outcome/vote_analysis.py', new_file_name], check=True)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python run_voting.py [repetitions] [model_name]")
        sys.exit(1)

    repetitions = int(sys.argv[1])
    model_name = sys.argv[2]
    run_voting_and_analysis(repetitions, model_name)
