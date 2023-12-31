# Conversation Machine Project

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Welcome to the Conversation Machine Project, a Python-based initiative designed to cultivate engaging and dynamic conversations through multi-agent intelligent topic suggestions.

At the heart of this project is a sophisticated algorithm that leverages natural language processing (NLP) techniques to analyze recent text exchanges in a conversation and propose new topics that are relevant and interesting. The goal is to foster deeper, more meaningful conversations by suggesting topics that are aligned with the current discussion, yet offer a fresh perspective or a deeper dive into the existing themes.

Utilizing techniques such as TF-IDF for keyword extraction and synonym expansion through WordNet, the Conversation Machine Project aims to enhance the quality of digital conversations, making them as enriching and engaging as face-to-face interactions.

Whether it's enhancing a chatbot's capabilities or aiding human users in finding the right words to keep the conversation flowing, the Conversation Machine Project is here to assist. It is an ongoing project, continuously evolving to incorporate more advanced features and offer a refined user experience. Join us in this endeavor to take digital conversations to the next level.

![Conversation Machine Image](conversation-machine.png)

## Features

The Conversation Machine Project comes packed with a range of features designed to enhance the depth and flow of digital conversations. Here, we delineate the key features that make this project stand out:

- **Dynamic Topic Suggestions**: Leveraging the power of NLP, the Conversation Machine Project can analyze the most recent exchanges in a conversation and suggest new topics that are both relevant and engaging. This feature aids in keeping the conversation flowing naturally, avoiding stale or repetitive discussions.
- **Keyword Extraction and Expansion**: By utilizing TF-IDF for keyword extraction and WordNet for synonym expansion, the Conversation Machine Project is equipped to delve deeper into the conversation's context, extracting the essence of the discourse and expanding upon it to generate rich and varied topic suggestions.
- **Relevance and Coherence Scoring**: To ensure the suggested topics are well-aligned with the ongoing conversation, this feature employs a multi-metric scoring system. It evaluates the relevance and coherence of potential topics, helping to prioritize suggestions that are most likely to foster meaningful and captivating discussions.

Feel free to further explore these features and discover how the Conversation Machine Project can revolutionize digital conversations, providing a more natural and enriching communicative experience.

## Installation

In this section, we guide you through the necessary steps to install and set up the Conversation Machine Project environment on your local system. Here is everything you need to get started:

### Prerequisites

Ensure that your system meets the following prerequisites before proceeding with the installation:

- **Python 3.x**: The Conversation Machine Project is developed using Python 3. Make sure to have a compatible Python version installed. You can download it from the [official Python website](https://www.python.org/).
- **NLTK package**: This project utilizes the Natural Language Toolkit (NLTK) for various natural language processing tasks. Install it using the following command:
```
pip install nltk
```
Spacy: Spacy is another essential NLP library used in this project. Install it with the following command:
```
pip install spacy
```
scikit-learn: The project employs scikit-learn for machine learning tasks, including text vectorization. Install it using the command:
```
pip install scikit-learn
```
Hugging Face: To take advantage of the state-of-the-art NLP models, ensure to have the Hugging Face library installed using:
```
pip install transformers
```
Other Dependencies: Depending on the specific functionalities of your project, you might require additional libraries and packages. Be sure to list them here.
Installation Steps
Clone the Repository: Start by cloning the Conversation Machine Project repository from GitHub to your local system using the following command:
```
git clone https://github.com/dolvido/conversation_machine.git
```
Navigate to the Project Directory: Change your directory to the project's root folder using:
```
cd conversation_machine
```
Install Dependencies: Install all the necessary dependencies using the command:
```
pip install -r requirements.txt
```
Run the Application: Finally, run the application using the following command:
```
python main.py
```
You have now successfully installed and set up the Conversation Machine Project on your system. You can start exploring its functionalities and features.
