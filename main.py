import os
import re
from dotenv import load_dotenv, find_dotenv
from itertools import combinations
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import nltk
from nltk.corpus import wordnet
import spacy

nlp = spacy.load("en_core_web_sm")

nltk.download('wordnet')


import pyttsx3

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


class EnvironmentLoader:
    def __init__(self):
        load_dotenv(find_dotenv())

    @property
    def huggingfacehub_api_token(self):
        return os.environ["HUGGINGFACEHUB_API_TOKEN"]


class DataPreprocessor:
    @staticmethod
    def preprocess_data(text):
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()

        # Remove punctuation using regex and then tokenize the words
        words = word_tokenize(re.sub(r'[^\w\s]', '', text.lower()))

        # Remove stopwords and apply stemming
        words = [ps.stem(word) for word in words if word not in stop_words]
        return words

class TopicModel:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.lda = LatentDirichletAllocation(n_components=3, random_state=42)
            
    def fit_transform(self, conversation_history):
        # Join all the conversation texts and vectorize them
        texts = [' '.join([turn['agent1'], turn['agent2'], turn['agent3']]) for turn in conversation_history]
        dtm = self.vectorizer.fit_transform(texts)
            
        # Fit the LDA model
        self.lda.fit(dtm)
            
        # Get the topics
        topics = self.lda.transform(dtm)
        return topics
        
    def get_most_common_topics(self):
        # Get the words associated with each topic
        topics = {}
        for index, topic in enumerate(self.lda.components_):
            words = [self.vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-3:]]
            topics[index] = words
        return topics


class Agent:
    def __init__(self, repo_id, template, input_variables, model_kwargs_dict):
        self.hub = HuggingFaceHub(repo_id=repo_id, model_kwargs=model_kwargs_dict)
        self.prompt = PromptTemplate(template=template, input_variables=input_variables)
        self.chain = LLMChain(prompt=self.prompt, llm=self.hub, memory=ConversationBufferMemory())

    def run(self, input_data):
        if input_data.strip():
            return self.chain.run(input_data)
        else:
            return "I don't have enough information to provide a meaningful response."


class ConversationManager:
    def __init__(self, agent1, agent2, agent3):
        self.agent1 = agent1
        self.agent2 = agent2
        self.agent3 = agent3
        self.conversation_history = []

    def get_conversation_text(self):
        """Get all the text from the conversation history as a single string."""
        texts = [' '.join([turn['agent1'], turn['agent2'], turn['agent3']]) for turn in self.conversation_history]
        return ' '.join(texts)

    def get_novelty_score(self, topic):
        # Get the recent topics from the conversation history
        recent_topics = [turn['agent3'] for turn in self.conversation_history[-3:]]  # Adjust the range as needed
    
        # Compute the similarity of the new topic with each recent topic
        similarities = [self.compute_similarity(topic, recent_topic) for recent_topic in recent_topics]
    
        # Use the inverse of the average similarity as the novelty score
        avg_similarity = sum(similarities) / len(similarities)
        novelty_score = 1 / (avg_similarity + 1e-5)  # Adding a small value to avoid division by zero
    
        return novelty_score
    
    def get_coherence_score(self, topic):
        # Split the topic into individual words
        words = topic.split()
        
        # Compute the pairwise similarity between the words
        similarities = [self.compute_similarity(words[i], words[j]) for i in range(len(words)) for j in range(i+1, len(words))]
        
        # Use the average pairwise similarity as the coherence score
        coherence_score = sum(similarities) / len(similarities) if similarities else 0.0
        
        return coherence_score

    def compute_similarity(self, text1, text2):
        # This is a placeholder function; you'll need to implement a function to compute the similarity between two texts
        return 0.5  # Placeholder value

    def get_relevance_score(self, topic):
        # Metric 1: Frequency in conversation history
        frequency_score = sum(self.get_conversation_text().count(word) for word in topic.split())
        
        # Metric 2: Novelty
        novelty_score = self.get_novelty_score(topic)
        
        # Metric 3: Coherence
        coherence_score = self.get_coherence_score(topic)
        
        # Combine the metrics to calculate the final relevance score
        relevance_score = frequency_score + novelty_score + coherence_score
        return relevance_score
    
    def generate_candidate_topics(self):
        # Get the recent texts from the conversation history
        recent_texts = [turn['agent1'] + " " + turn['agent2'] for turn in self.conversation_history[-2:]]
    
        # Extract keywords from the recent texts
        keywords = self.extract_keywords(recent_texts)
    
        # Flatten the list of lists into a single list
        flat_keywords = [item for sublist in keywords for item in sublist]
    
        # Generate expanded keywords for each keyword
        expanded_keywords = [self.expand_keyword(keyword) for keyword in flat_keywords]

        # Step 3: Formulate candidate topics by combining the keywords and expanded keywords
        candidate_topics = self.formulate_topics(keywords, expanded_keywords)

        return candidate_topics

    def extract_keywords(self, recent_texts):
        keywords = []

        for text in recent_texts:
            # Process the text using the spaCy pipeline
            doc = nlp(text)

            # Extract noun phrases as keywords
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
                
            # Here, we take top N noun phrases based on their length
            N = 3
            noun_phrases = sorted(noun_phrases, key=len, reverse=True)[:N]
                
            # Add the list of noun phrases for this text to the keywords list
            keywords.append(noun_phrases)

        return keywords

    def expand_keyword(self, keyword):
        # Find synonyms using WordNet
        synonyms = []
        for syn in wordnet.synsets(keyword):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())

        # If synonyms are found, use the first two as related keywords
        if synonyms:
            return synonyms[:2]
        else:
            return [keyword, keyword]  # Repeat the keyword to have at least two elements


    def formulate_topics(self, keywords, expanded_keywords):
        # Generating combinations of keywords and expanded keywords to create candidate topics
        candidate_topics = []
        for keyword in keywords:
            for expanded_keyword in expanded_keywords:
                # Ensure keyword and expanded_keyword are strings
                keyword_str = ' '.join(keyword) if isinstance(keyword, list) else keyword
                
                candidate_topics.append(f"{keyword_str} {expanded_keyword[0]}")
                candidate_topics.append(f"{keyword_str} {expanded_keyword[1]}?")
        
        # Adding some combinations of keywords
        for combo in combinations(keywords, 2):
            # Ensure combo elements are strings
            combo_str_0 = ' '.join(combo[0]) if isinstance(combo[0], list) else combo[0]
            combo_str_1 = ' '.join(combo[1]) if isinstance(combo[1], list) else combo[1]
            
            candidate_topics.append(f"{combo_str_0} {combo_str_1}?")
        #print(candidate_topics)
        return candidate_topics

    def suggest_topic(self):
        # Generate candidate topics
        candidate_topics = self.generate_candidate_topics()

        # Evaluate each candidate topic using the relevance score function
        topic_scores = {topic: self.get_relevance_score(topic) for topic in candidate_topics}

        # Select the topic with the highest relevance score

        if not topic_scores:  # Check if the topic_scores dictionary is empty
            return "It seems we have covered a lot of interesting topics. Do you have any other questions or topics in mind?"
    
        best_topic = max(topic_scores, key=topic_scores.get)
        return "What do you think about the concept of " + best_topic + "?"

    
    def run(self, initial_question, max_cycles=2):
        question = initial_question

        while True:
            # Agent 1 asks a question
            response_agent_1 = self.agent1.run(question)
            print("Agent 1: ", response_agent_1)
            #speak_text(response_agent_1)
                
            # Agent 2 responds to the question
            response_agent_2 = self.agent2.run(response_agent_1)
            print("Agent 2: ", response_agent_2)
            #speak_text(response_agent_2)

            # Agent 3 proposes a new topic based on the previous conversation
            response_agent_3 = self.agent3.run(response_agent_1.strip() + " " + response_agent_2.strip())
            print("Agent 3: ", response_agent_3)
            #speak_text(response_agent_3)

            # Adding the conversation to the history
            self.conversation_history.append({
                'agent1': response_agent_1, 
                'agent2': response_agent_2, 
                'agent3': response_agent_3
            })


            # Suggesting a new topic based on the conversation history
            new_topic = self.suggest_topic()
            #print("New topic: ", new_topic)
            #speak_text(new_topic)
            new_topic = response_agent_3
            # Setting the new topic as the next question for agent 1
            question = new_topic
            # Breaking the loop after reaching the maximum number of cycles
            if len(self.conversation_history) >= max_cycles:
                break


# Loading environment variables
env_loader = EnvironmentLoader()
HUGGINGFACEHUB_API_TOKEN = env_loader.huggingfacehub_api_token

# Initializing agents with dictionary to store model kwargs
common_model_kwargs = {"temperature": 0.1, "max_new_tokens": 500}
agent1 = Agent(
    repo_id="tiiuae/falcon-7b-instruct", 
    template="Question: {question}", 
    input_variables=["question"], 
    model_kwargs_dict={"temperature": 0.1, "max_new_tokens": 500}
)
agent2 = Agent(
    repo_id="tiiuae/falcon-7b-instruct", 
    template="Can you elaborate or provide your perspective on the following statement: {answer}",
    input_variables=["answer"], 
    model_kwargs_dict=common_model_kwargs
)
agent3 = Agent(
    repo_id="tiiuae/falcon-7b-instruct", 
    template="What is a good follow up topic that reminds you of: {topic}", 
    input_variables=["topic"], 
    model_kwargs_dict={"temperature": 0.9, "max_new_tokens": 120}
)

# Initializing and running conversation manager
conv_manager = ConversationManager(agent1, agent2, agent3)
conv_manager.run(initial_question="What is the meaning of life?", max_cycles=6)
