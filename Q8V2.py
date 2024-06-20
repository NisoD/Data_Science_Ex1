import numpy as np
import json
import os
import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pprint

def preprocess(text):
    """Function for cleaning Wikipedia data"""
    text = re.sub(r'\[\d+\]', '', text)  # Remove reference markers like [1], [2]
    text = ' '.join(text.split())  # Remove extra whitespace
    text = re.sub(r'[^\w\s,.()\'%-]', '', text)  # Remove non-alphanumeric characters except punctuation
    return text

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

def build_similarity_matrix(sentences):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(tfidf_matrix)
    return sim_matrix

def calculate_page_rank(sim_matrix, d=0.85, eps=1.0e-8, max_iter=100):
    n = sim_matrix.shape[0]
    matrix = d * sim_matrix + (1 - d) / n * np.ones([n, n])
    ranks = np.ones(n) / n

    for _ in range(max_iter):
        new_ranks = matrix @ ranks
        new_ranks /= np.linalg.norm(new_ranks, 1)  # Normalize to prevent overflow
        if np.linalg.norm(ranks - new_ranks, 1) < eps:
            break
        ranks = new_ranks
    return ranks

def summarize_text(text, num_sentences=5):
    sentences = text.split('. ')
    if len(sentences) > 100:  # Limit to a reasonable number of sentences
        sentences = sentences[:100]
    
    sim_matrix = build_similarity_matrix(sentences)
    ranks = calculate_page_rank(sim_matrix)
    ranked_sentences = [sentences[i] for i in np.argsort(ranks)[::-1]]
    summary = '. '.join(ranked_sentences[:num_sentences])
    summary = re.sub(r'\s*\.\s*', '. ', summary)  # Ensure proper spacing around periods
    summary = summary.replace('..', '.')
    return summary

def summarize_text_per_fruit(fruit_texts):
    summaries = {}
    for fruit, text in fruit_texts.items():
        summary = summarize_text(text)
        summaries[fruit] = summary
    pprint.pprint(summaries)
    return summaries

if __name__ == "__main__":
    # fruitcrawl()  # Uncomment if you need to crawl the data again
    fruit_texts = load_fruit_texts('fruit_texts')
    summarize_text_per_fruit(fruit_texts)
