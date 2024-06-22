import numpy as np
import json
import os
import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import pprint
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from collections import Counter
from Q7 import tdidf




def get_wikipedia_text(fruit):
    url = f"https://en.wikipedia.org/wiki/{fruit}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')

    fruit_text = ""
    for p in paragraphs:
        fruit_text += p.text+('\n')

    return fruit_text.strip()


def preprocess_for_summery(text):
    # Remove references like [7], [31], etc.
    text = re.sub(r'\[\d+\]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Split text into sentences
    return text.split(". ")


def fruitcrawl():
    fruits = ["Apple", "Banana", "Cherry", "Date_palm", "Grape", "Orange_(fruit)", "Peach", "Pear",
        "Plum", "Watermelon", "Blueberry", "Strawberry", "Mango", "Kiwifruit", "Papaya",
        "Pineapple", "Lemon", "Lime_(fruit)", "Raspberry", "Blackberry"]

    if not os.path.exists('fruit_texts'):
        os.makedirs('fruit_texts')

    for fruit in fruits:
        print(f"Crawling wiki page for {fruit}...")
        text = get_wikipedia_text(fruit)
        if text:
            # text = preprocess(text, fruit)  # Pass the fruit name for specific preprocessing
            with open(f'fruit_texts/{fruit}.json', 'w', encoding='utf-8') as f:
                json.dump({"fruit": fruit, "text": text}, f, ensure_ascii=False, indent=4)
        else:
            print(f"Could not retrieve text for {fruit}")


def retrive_data(q:str):
    if q == 'b':
        fruit_texts = {}
        for filename in os.listdir('fruit_texts'):
            if filename.endswith(".json"):
                with open(f'fruit_texts\\{filename}', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    fruit_texts[data["fruit"]] = data["text"]
                f.close()
        return fruit_texts
    
    if q=='c':
        fruit_sum = {}
        for filename in os.listdir('fruit_summaries'):
            if filename.endswith(".json"):
                with open(f'fruit_summaries\\{filename}', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    fruit_sum[data["fruit"]] = data["summary"]
                f.close()
        return fruit_sum


# def calculate_td(doc:str) ->dict:
#     td = Counter(doc.split())
#     return {word: count for word, count in td.items()}


# def calculate_idf(documents:list) -> dict:
#     idf = {}
#     num_of_docs = len(documents)
#     all_terms = set(word for doc in documents for word in doc.split())
#     for term in all_terms:
#         doc_count_with_term = sum(1 for doc in documents if term in doc.split())
#         if doc_count_with_term == 0:
#             idf[term] = 0
#         else:
#             idf[term] = np.log10(num_of_docs / doc_count_with_term)
#     return idf


# def td_idf(doc:str, idf):
    # td:dict = calculate_td(doc)
    # return {term: td_val * idf.get(term, 0) for term, td_val in td.items()}


def build_similarity_matrix(sentences):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return sim_matrix


def calculate_page_rank(sim_matrix, beta=0.85, eps=1.0e-8, iterations=10000):
    sim_matrix = np.array(sim_matrix)
    n = sim_matrix.shape[0]
    ranks = np.ones(n) / n
    
    A = beta * sim_matrix + (1 - beta) / n * np.ones([n, n])
    
    for _ in range(iterations):
        new_ranks = A @ ranks        
        if np.linalg.norm(ranks - new_ranks, 1) < eps:
            return new_ranks
        ranks = new_ranks
    return ranks


def summarize_text(text:str):
    """text is the text per frruit"""
    sentences:list = preprocess_for_summery(text)
    sim_matrix = build_similarity_matrix(sentences)
    page_rank = calculate_page_rank(sim_matrix)
    summary = [sentence for sentence, rank in sorted(zip(sentences, page_rank), key=lambda x: x[1], reverse=True)[:5]]
    return summary


def summarize_text_per_fruit(fruits_texts):
    summaries = {}
    if not os.path.exists("fruit_summaries"):
        os.makedirs("fruit_summaries")
    
    for fruit, text in fruits_texts.items():
        summaries[fruit] = summarize_text(text)
        with open(f'fruit_summaries\\{fruit}_summary.json', 'w', encoding='utf-8') as f:
            json.dump({"fruit": fruit, "summary": summaries[fruit]}, f, ensure_ascii=False, indent=4)
    
    return summaries
    

def textsum():
    full_text:dict = retrive_data('b')
    summarize_text_per_fruit(full_text)
    

def q8_c():
    full_summaries:dict = retrive_data('c')
    for fruit, summery in full_summaries.items():
        str_summery = " ".join(summery)
        uniqe_w = set(str_summery.replace('-','').replace(',','').replace('(','').replace(')','').split())
        score = tdidf(str_summery,uniqe_w)
        print(score)





def calculate_e_distance(centeroid:list , data_point):
    #Euclidean distance
    return np.linalg.norm(np.array(data_point) - np.array(centeroid))


#THE K-MEANS ALGORITHM
def calculate_kmeans(feature_vectors:list, k:int, iteration:int ):
    num_of_data_points = len(feature_vectors)
    centroids = feature_vectors[:k]
    data_point_groups = np.zeros(num_of_data_points)

    for i in range(iteration):

        for n in range(num_of_data_points):
            distances = [calculate_e_distance(centroid, feature_vectors[n]) for centroid in centroids]
            data_point_groups[n] = np.argmin(distances)
        
        if i != iteration - 1:
            for j in range(k):
                cluster_points = [feature_vectors[index] for index in range(num_of_data_points) if data_point_groups[index] == j]
                if cluster_points:
                    centroids[j] = np.mean(cluster_points, axis=0)


    
    return data_point_groups,centroids


def kmeans():
    a,b,c = "Amount of Sugar","Time it Lasts","Price"
    feature_vectors = []
    for _ , row in DataFrame.iterrows():
        feature_vectors.append(np.array([row[a], row[b], row[c]]))

    data_point_groups,centroids  = calculate_kmeans(feature_vectors,4,40)

    x_point = [v[0] for v in feature_vectors]
    y_point = [v[2] for v in feature_vectors]
    centroids_x =[c[0] for c in centroids]
    centroids_y = [c[2] for c in centroids]

    fig = make_subplots(rows=1, cols=1)
    fig.update_layout(title_text="qustion 8 d ")
    fig.update_xaxes(title_text="Amount of Sugar")
    fig.update_yaxes(title_text="Price")

    fig.add_trace(go.Scatter( x=x_point, y=y_point,name="Data Points",mode="markers", marker=dict(size=13, color=data_point_groups)))
    fig.add_trace(go.Scatter(x=centroids_x,y=centroids_y,name="Centeroids",mode="markers",marker=dict(symbol="x", size=13)))

    fig.show()
    

def q8_e():
    Peeling_Messiness = {'Low':0, 'Medium':0.5, 'High':1}
    seasons = {"Winter": 0,'Fall':0.25 ,"Spring": 0.75 , "Summer": 1} #valued simillarity
    colors = ['Yellow','Green','Pink', 'Orange','Red', 'Purple', 'Blue', 'Brown', 'Black'] #sorted by brightness

    a,b,c = "Peeling/Messiness", "Color", "Growth Season"

    feature_vectors = []
    for _ , row in DataFrame.iterrows():
        feature_vectors.append(np.array([Peeling_Messiness[row[a]], colors.index(row[b])/len(colors), seasons[row[c]]]))

    
    data_point_groups, _ = calculate_kmeans(feature_vectors,4,40)
    x_point = DataFrame["Amount of Sugar"].tolist()
    y_point = DataFrame["Price"].tolist()

    fig = make_subplots(rows=1, cols=1)
    fig.update_layout(title_text="qustion 8 e ")
    fig.update_xaxes(title_text="Amount of Sugar")
    fig.update_yaxes(title_text="Price")

    fig.add_trace(go.Scatter(x=x_point, y=y_point,name="Data Points",mode="markers", marker=dict(size=13, color=data_point_groups)))
    fig.show()
    

if __name__ == "__main__":
    global DataFrame
    DataFrame = pd.read_csv('fruits.csv')
    # fruitcrawl() # q_a
    # textsum() #q_b
    q8_c()
    # kmeans() #q_d
    # q8_e()


# def build_similarity_matrix(sentences:list):
#     idf:dict = calculate_idf(sentences)
#     sim_matrix = []
#     tfidf_all_s = [td_idf(s, idf) for s in sentences]

#     for i in range(len(sentences)):
#         row = []
#         for j in range(len(sentences)):
#             if i == j:
#                 row.append(1.0)  
#                 continue
#             doc1_tfidf = tfidf_all_s[i]
#             doc2_tfidf = tfidf_all_s[j]
#             dot_product = sum(tfidf1 * tfidf2 for tfidf1, tfidf2 in zip(doc1_tfidf.values(), doc2_tfidf.values()))
#             magnitude1 = np.linalg.norm(list(doc1_tfidf.values()))
#             magnitude2 = np.linalg.norm(list(doc2_tfidf.values()))
#             if magnitude1 == 0 or magnitude2 == 0:
#                 row.append(0.0)  
#             else:
#                 cosine_similarity = dot_product / (magnitude1 * magnitude2)
#                 row.append(cosine_similarity)
#         sim_matrix.append(row)

#     return sim_matrix