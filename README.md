# LLM voting

## Abstract
This paper investigates the voting behaviors of Large Language Models (LLMs), specifically GPT-4 and LLaMA-2, their biases, and how they align with human voting patterns. Our methodology involved using a dataset from a human voting experiment to establish a baseline for human preferences and a corresponding experiment with LLM agents. We observed that the methods used for voting input and the presentation of choices influence LLM voting behavior. We discovered that varying the persona can reduce some of these biases and enhance alignment with human choices. While the Chain-of-Thought approach did not improve prediction accuracy, it has potential for AI explainability in the voting process. We also identified a trade-off between preference diversity and alignment accuracy in LLMs, influenced by different temperature settings. Our findings indicate that LLMs may lead to less diverse collective outcomes and biased assumptions when used in voting scenarios, emphasizing the importance of cautious integration of LLMs into democratic processes.

## Citing
If you found any part of this work useful, you are strongly encouraged to cite:
```
@misc{yang2024llmvoting,
  doi = {10.48550/ARXIV.2402.01766},
  url = {https://arxiv.org/abs/2402.01766},
  author = {Yang,  Joshua C. and Dailisan,  Damian and Korecki,  Marcin and Hausladen,  Carina I. and Helbing,  Dirk},
  keywords = {Computation and Language (cs.CL),  Artificial Intelligence (cs.AI),  Computers and Society (cs.CY),  Machine Learning (cs.LG),  General Economics (econ.GN),  FOS: Computer and information sciences,  FOS: Computer and information sciences,  FOS: Economics and business,  FOS: Economics and business,  I.2.7; J.4; K.4.1,  68T05,  91B14,  91C20},
  title = {LLM Voting: Human Choices and AI Collective Decision Making},
  publisher = {arXiv},
  year = {2024},
  copyright = {arXiv.org perpetual,  non-exclusive license}
}
```
## quick start
create a new python environment (3.9 recommended) and install from the requirements file. 


