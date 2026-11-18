# NLP First-experience Lab

**Hands-on First Experience of Natural Language Processing**

## Homework 5

- Release date: 2026/11/19
- Due date: 2026/12/01 21:00
  - Note: Lab 2.2 has a different due date: **2026/12/13 21:00**

In this assignment, we hope that you will become familiar with prompts and be able to utilize the API of a Large Language Model to solve Natural Language Processing tasks.

TA:
- 李立庠 b12611001 [at] ntu.edu.tw
- 許安彤 b12611005 [at] ntu.edu.tw

Previously designed by 賴鴻昇. Original idea suggested by 李朋軒.

tags: `1151iai`

Authors:
- [Li Peng-Hsuan 李朋軒](https://jacobvsdanniel.github.io) (jacobvsdanniel [at] gmail.com)
- Lai Hong-Sheng 賴鴻昇 (b08611048 [at] ntu.edu.tw)

## Overview

- Solve Natural Language Processing (NLP) tasks
- Apply Large Language Models (LLM) with prompting methods
- Intended for beginners
  - The source codes are short: be expected to understand them fully.
  - Be expected to develop your own program.
  - No GPU requirements or machine learning model engineering: we'll be using online services.

## Environment

- Operating system
  - OS-independent
- System apps
  - [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.anaconda.com/miniconda/)
  - Python 3.11 (created by conda below)
- Python packages
  - See *requirements.txt*
  - This homework uses the legacy `openai==0.28.1` SDK. **Do not upgrade the `openai` package** (e.g. `pip install -U openai`), or the code will break.

> Windows users: run the commands below in **Anaconda Prompt** (Start menu → Anaconda Prompt), not the default Command Prompt or PowerShell, otherwise `conda` may not be found.

```bash
# move to your working directory
cd /path/to/your/working/directory

# Download this repository (If you don't have git, download and unzip the zip file instead.)
git clone https://github.com/antunghsu/nlp_lab
cd nlp_lab

# Create and activate a conda environment with Python 3.11
conda create -n nlp_lab python=3.11
conda activate nlp_lab

# To check python version (should be 3.11.x)
python --version

# Install the required packages with PyPI (recommended), or
pip install -r requirements.txt
# Install with Conda (from the conda-forge channel)
conda install -c conda-forge --file requirements.txt
```


## Lab 1

Prompting Large Language Models (LLM) to solve Natural Language Processing (NLP) tasks

### Lab 1.0. Preparation

#### Step 1. These are the relevant files:

- *lab1_nlp_tasks.py*
- *lab1_config.json*
- *corpora/news/*
- *corpora/prompt/*

#### Step 2. These are the core concepts:
- Tasks
  - Named Entity Recognition (NER)
  - Relation Extraction (REL)
  - Summarization (SUM)
- LLM prompting methods
  - Instruction prompting
  - In-context learning
- Languages
  - English (en)
  - Chinese (zh)

#### Step 3. [Create your OpenAI API Key](https://platform.openai.com/account/api-keys/)

It is a string that looks like "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx".
If you have already filled in the Google form, you should be able to see your organization: "Intro of AI".
We already fund the API for you, so you don't need to pay for it. But please don't abuse it.

> Note: Please remember to keep your API key. It only shows once. And do not share it with others.

#### Step 4. Check *lab1_config.json*

Modify it as you wish.

#### Step 5. Run *lab1_nlp_tasks.py*

For example, in your terminal (Anaconda Prompt on Windows) with the `nlp_lab` environment activated, run
```bash
python lab1_nlp_tasks.py
```

#### Step 6. Check results in *lab1_output/*

For example, in *lab1_output/ner__instruction__en__Russo-Ukrainian_War.txt*:
```
Extract entities of persons, locations, organizations from the article.

Article:
The Russo-Ukrainian War is an ongoing international conflict...

####################################################################################################
# OUTPUT
####################################################################################################

1. Viktor Yanukovych (person)
...
```

The results may vary for each different runs.

### Lab 1.1. Solve existing tasks

#### Objectives

Solve all tasks (NER, REL, SUM) in both languages (English, Chinese) with both prompting methods (instruction prompting, in-context learning).
> The English prompts is already there; you only need to get output results. And create your own Chinese prompts.
> If you have question about Chinese prompts, please contact TA.

#### Scoring
One task in one language: 5 point.

3 tasks in 2 languages: 30 point.

### Lab 1.2. Solve new tasks

#### Objectives

Solve tasks beyond NER, REL, SUM
- For at least 2 tasks, For inspiration:
  - [CKIP CoreNLP](https://ckip.iis.sinica.edu.tw/service/corenlp/)
  - [Stanford CoreNLP](https://corenlp.run/)
  - [Hugging Face Tasks](https://huggingface.co/tasks)
  - [LDC Catalog](https://catalog.ldc.upenn.edu/)
- In at least 2 languages
  - One of them must be English
  - The other may be Chinese or another language
- You can use other news materials, add them in corresponding folders, and don't need to provide them in your submission.

#### Scoring
One task in one language: 10 point.

2 tasks in 2 languages: 40 point.


## Lab 2

Using LLM train of thought to achieve complicated behavior

### Lab 2.0. Preparation

#### Step 1. These are the relevant files:

- *lab2_train_of_thought.py*
- *lab2_config.json*

#### Step 2. These are the core concepts:

- Parse LLM output to structured data
  - To use them programmatically beyond chatting with humans.
- Make LLM digest its own outputs
  - To get explanations, spot errors, edit results, do follow-up tasks, etc.

### Lab 2.1. A Druid's Journey

#### Objectives

Play the **修行的德魯伊(A Druid's Journey, 2023)** game
- Run *lab2_train_of_thought.py*
- Achieve the game objectives
- Check results in *lab2_output/* (you should see at least *summary_1.txt*)

#### Scoring
Finish the game: 10 point.

### Submission for Lab 1 + Lab 2.1

```
{student_id}_hw5    # In lowercase
├── lab1_output
|   ├── ner__in-context_learning__en__2023_Hawaii_wildfires.txt
|   ├── ner__in-context_learning__en__Russo-Ukrainian_War.txt
|   └── ...
├── lab1_prompt # You can copy from `corpora/prompt`
|   ├── ner__in-context_learning__en.txt
|   ├── ner__in-context_learning__zh.txt
|   └── ...
└── lab2_output
    ├── state
    |   └── save_1.json
    ├── summary_1.txt
    └── ...

# Zip it and submit to NTU COOL
-> {student_id}_hw5.zip
```

### Lab 2.2. Your own game
This lab takes a lot of time (at least 10 hours), so you can submit it before 2026/12/13 21:00.
I believe you can learn a lot from this lab, so I highly recommend you to try it.
The TA will give you feedback on your game.
You can have bonus points depending on the quality of your game.

#### Objectives

Create your own game
- It may be a text adventure game or be in other forms of your wish.
- It must use multiple LLM tasks.
- It must require LLM to digest its own outputs.
- It must create an enjoyable review for each game run.

#### Scoring
40 point. (We will select up to three students with the best overall performance to receive a two-point bonus on your final grade.)

#### Submission
Please submit your assignment on NTU COOL by providing a link to your GitHub repository or Google Drive.

- Your game
  - Program
  - Data
  - Manual (if needed)
- Runs of your game
  - The final game states and output files of at least one full run
- A report
  - Game description
  - Details on the LLM tasks applied
  - Your discovery

## LICENSE

Copyright © 2023 [Li Peng-Hsuan 李朋軒](https://jacobvsdanniel.github.io) Lai Hong-Sheng 賴鴻昇

Modified in 2026 by the course TA: updated the default model to `gpt-4o-mini` (`gpt-3.5-turbo` / `gpt-4` replaced), updated `tiktoken` to 0.7.0, and switched environment setup to conda with Python 3.11. Original repository: <https://github.com/wkh1267/nlp_lab>.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
