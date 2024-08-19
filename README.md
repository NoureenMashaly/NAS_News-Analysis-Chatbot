# NAS: A news analysis system with a chatbot interface for news consumption
## Overview
Using several deep learning and NLP techniques such as Transformers, data preprocessing, webscarping, dialog flow, flask and fullfilment, this project aims to streamline news consumption by embedding deep learning and machine learning models to analyze news articles before they reach users. 
The system
<ul>
  <li> categorizes articles by topic </li> 
  <li> highlights breaking news </li>
  <li> generates informative headlines </li>
  <li> provides concise summaries of key points </li> 
  This approach helps users quickly grasp essential information without reading lengthy articles.

# Features
![image](https://github.com/user-attachments/assets/34ff8264-fcc5-44c3-8ac0-323d5683ba6f)

## Topic Modeling
<ul>
  <li>Objective:Identify and cluster similar words to categorize articles by topic.</li>
  <li>Technologies Used: Llama2, HDBSCAN, UMAP, SBERT, C-TI-IDF, CountVectorizer, and prompt engineering.</li>
</ul>

## Breaking News Classification
<ul>
  <li>Objective: Highlight important, real-time events.</li>
  <li>Technologies Used: Finetuned BERT using TensorFlow, TensorFlow Hub, TensorFlow Text, BERT, and custom preprocessing and classification layers.</li>
</ul>

## Model Performance
Breaking News Classification: Achieved high accuracy and low loss during training and evaluation.
<ul>
  <li> Training Results: Loss: 0.0318, Accuracy: 0.9917</li>
  <li>Validation Results: Loss: 0.2779, Accuracy: 0.9356</li>
  <li>Evaluation Results: Loss: 0.2026, Accuracy: 0.9442</li>
</ul>

## Dataset
<ul>
  <li>Creation: The custom dataset used for training the breaking news classification model was created using web scraping techniques.</li>
  <li>Web Scraping: Employed Beautiful Soup for extracting data from various news websites.</li>
  <li>Kaggle: The dataset is available on Kaggle. https://www.kaggle.com/datasets/yomnamuhammad/breaking-news</li>
</ul>

## Headline Generation
<ul>
  <li> Objective: Create compelling and informative titles for each article.</li>
  <li>Technologies Used: Llama2, prompt engineering, 4-bit quantization, and Hugging Face transformers.
</li>
</ul>

## Text Summarization
<ul>
  <li>Objective: Provide concise summaries of key points.</li>
  <li>Technologies Used: BERT extractive summarizer (Summarizer), transformers, and Hugging Face.</li>
</ul>

# Chatbot Integration
<ul>
  <li>Objective: Streamline news delivery to users through a friendly conversational interface.</li>
  <li>Technologies Used: Dialogflow, Telegram Flask, ngrok, and Dialogflow’s fulfillment logic.</li>
  <li>Functionality: The chatbot streams five news articles at a time, each processed by the topic modeling, breaking news classification, headline generation, and text summarization models, and presents the        results to the user.</li>
  <li>Knowledge Base: Utilized Dialogflow’s knowledge base feature to allow users to ask about famous individuals and receive relevant information.</li>
</ul>

# Methods and Techniques
## Embeddings and Contextual Understanding
Utilizes BERT’s contextual embeddings for more accurate and meaningful topic extraction compared to traditional methods like LDA.
## Dynamic Topic Discovery
Employs HDBSCAN for discovering a varying number of topics, adapting to the underlying structure of the data.
## Flexibility and Adaptability
Adapts to different languages and domains by using appropriate pre-trained BERT models or other transformer-based embeddings.
## Usage
There are four main files, each corresponding to a different model, along with a fulfillment logic script. Download all four model files and the fulfillment logic script. Update the paths in the ModelIsLoading function within the fulfillment script to point to the locations of your downloaded models. Finally, execute the cells until the Flask server starts.
