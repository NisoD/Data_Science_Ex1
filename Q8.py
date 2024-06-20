import numpy as np
import networkx as nx
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
import requests
from bs4 import BeautifulSoup
import re

def tf(descriptions, words):
    tf_words = {}
    for word in words:
        tf_word = []
        for doc in descriptions:
            tf_word.append(doc.count(word))
        tf_words[word] = tf_word
    return tf_words

def in_how_many_docs_appears(description, word):
    c = 0
    for doc in description:
        if word in doc:
            c += 1
    return c

def idf(num_of_docs, descriptions, words):
    idf_words = {}
    for word in words:
        word_count = in_how_many_docs_appears(descriptions, word)
        if word_count != 0:
            idf_words[word] = np.log10(num_of_docs / word_count)
        else:
            idf_words[word] = 0
    return idf_words

def compute_tfidf(tfs, idfs):
    tfidf_words = {}
    for word in idfs:
        l = []
        for n in tfs[word]:
            l.append(n * idfs[word])
        tfidf_words[word] = l
    return tfidf_words

def get_wikipedia_text(fruit):
    url = f"https://en.wikipedia.org/wiki/{fruit}"
    response = requests.get(url)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')

    fruit_text = ""
    for para in paragraphs:
        fruit_text += para.text
    return fruit_text.strip()

def preprocess(text):
    """Function for cleaning Wikipedia data"""
    text = re.sub(r'\[\d+\]', '', text)
    text = ' '.join(text.split())
    text = re.sub(r'[^\w\s,()\'%-]', '', text)
    return text

def fruitcrawl():
    fruits = [
        "Apple", "Banana", "Cherry", "Date_palm", "Grape", "Orange_(fruit)", "Peach", "Pear",
        "Plum", "Watermelon", "Blueberry", "Strawberry", "Mango", "Kiwifruit", "Papaya",
        "Pineapple", "Lemon", "Lime_(fruit)", "Raspberry", "Blackberry"
    ]

    if not os.path.exists('fruit_texts'):
        os.makedirs('fruit_texts')

    for fruit in fruits:
        print(f"Crawling wiki page for {fruit}...")
        text = get_wikipedia_text(fruit)
        if text:
            text = preprocess(text)
            with open(f'fruit_texts/{fruit}.json', 'w', encoding='utf-8') as f:
                json.dump({"fruit": fruit, "text": text}, f, ensure_ascii=False, indent=4)
        else:
            print(f"Could not retrieve text for {fruit}")

def load_fruit_texts(directory):
    fruit_texts = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                fruit_texts[data["fruit"]] = data["text"]
    return fruit_texts

def build_similarity_matrix(tfidf_matrix):
    cosine_similarities = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(cosine_similarities, 0)
    return cosine_similarities

def pagerank_summarization(text, num_sentences=5):
    sentences = text.split('. ')
    tf_descriptions = sentences
    words = list(set(" ".join(sentences).split()))
    
    tfs = tf(tf_descriptions, words)
    idfs = idf(len(tf_descriptions), tf_descriptions, words)
    tfidf_values = compute_tfidf(tfs, idfs)
    
    tfidf_matrix = pd.DataFrame(tfidf_values).values
    
    similarity_matrix = build_similarity_matrix(tfidf_matrix)
    
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = " ".join([s for _, s in ranked_sentences[:num_sentences]])
    return summary

def textsum():
    fruit_texts = load_fruit_texts('fruit_texts')
    
    summaries = {}
    for fruit, text in fruit_texts.items():
        summary = pagerank_summarization(text)
        summaries[fruit] = summary
        print(f"Summary for {fruit}:\n{summary}\n")
    
    return summaries

if __name__ == "__main__":
    fruitcrawl()
    textsum()
