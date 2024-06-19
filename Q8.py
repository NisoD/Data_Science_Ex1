import requests
from bs4 import BeautifulSoup
import json
import os
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

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
    """function for cleaning wikipedia data"""
    # remove [number]
    text = re.sub(r'\[\d+\]', '', text)
    # remove  whitespace and newlines
    text = ' '.join(text.split())
    
    # remove special characters
    text = re.sub(r'[^\w\s,.()\'%-]', '', text)
    
    return text

def fruitcrawl(): # as wikipedia didnt have specific patter we manually edited the names to match the pages
    fruits = [
        "Apple", "Banana", "Cherry", "Date palm", "Grape", "Orange_(fruit)", "Peach", "Pear",
        "Plum", "Watermelon", "Blueberry", "Strawberry", "Mango", "Kiwifruit", "Papaya",
        "Pineapple", "Lemon", "Lime (fruit)", "Raspberry", "Blackberry"
    ]

    if not os.path.exists('fruit_texts'):
        os.makedirs('fruit_texts')

    for fruit in fruits:
        print(f"Crawling wiki page for {fruit}...")
        text = get_wikipedia_text(fruit)
        text =preprocess(text)
        pprint.pprint(text)
        if text:
            with open(f'fruit_texts/{fruit}.json', 'w', encoding='utf-8') as f:
                json.dump({"fruit": fruit, "text": text}, f, ensure_ascii=False, indent=4)
            
        else:
            print(f"Could not retrieve text for {fruit}")

if __name__ == "__main__":
    fruitcrawl()
    