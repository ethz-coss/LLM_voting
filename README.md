# LLM Voting Repository

This GitHub repository hosts the source code and data for the research paper titled "LLM Voting: Human Choices and AI Collective Decision Making," which investigates the voting behaviors of Large Language Models (LLMs), specifically GPT-4 and LLaMA-2. The paper has been accepted for publication at the AAAI Conference on AI, Ethics, and Society (AIES).

## Repository Structure
- `agent/`: Python modules for LLM interaction.
- `data/`: Datasets including lab metadata, personas, and voting results.
- `figures/`: Generated figures and graphs for analysis.
- `llm/`: API and utilities for LLM configuration and communication.
- `memory/`: Modules for handling LLM memory and state information.
- `outcome/`: Results and analysis from various voting experiments.
   - `analyse_outcome.ipynb`: Jupyter notebook for data analysis and visualization of voting outcomes.
- `scripts/`: Python scripts for running the voting experiments.
- `EULER.md`: Computational resources documentation.
- `README.md`: This file, describing the project and setup instructions.
- `requirements.txt`: List of Python dependencies.

## Quick Start
1. **Setup Environment**:
   - Create a new Python environment using Python 3.9:
     ```bash
     python -m venv env
     source env/bin/activate
     ```
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```
2. **Environment Variables**:
   - Make sure the `OPENAI_API_KEY` is defined in the `.env` file in the root directory for API access.
3. **Run the Basic Voting Script**:
   - Navigate to the `scripts` folder and run the `pb_voting_basic.py`:
     ```bash
     cd scripts
     python pb_voting_basic.py
     ```
4. **Analyze the Outcomes**:
   - To analyze the results and generate visualizations, open the `analyse_outcome.ipynb` notebook located in the `outcome/` directory:
     ```bash
     jupyter notebook analyse_outcome.ipynb
     ```
     
## Citing Our Work
If you use this repository for your research, please cite our paper:
```bibtex
@inproceedings{yang2024llmvoting,
  title={LLM Voting: Human Choices and AI Collective Decision Making},
  author={Yang, Joshua C. and Dailisan, Damian and Korecki, Marcin and Hausladen, Carina I. and Helbing, Dirk},
  booktitle={AAAI Conference on AI, Ethics, and Society (AIES)},
  year={2024},
  doi={10.48550/arXiv.2402.01766},
  url={https://doi.org/10.48550/arXiv.2402.01766}
}

