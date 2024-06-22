import numpy as np
import json
import os
import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pprint
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from Q7 import tdidf
import nltk
from nltk.corpus import stopwords


def get_wikipedia_text(fruit):
    """ This function retrieves the text from the wikipedia page of the fruit."""
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


def preprocess_for_summary(text):
    """""Preprocess the text for summarization. This includes removing references and extra spaces."""
    
    text = re.sub(r'\[\d+\]', '', text)     # Remove references like [7], [31], etc.
    text = re.sub(r'\s+', ' ', text).strip()     # Remove extra spaces
    return text.split(". ")     # Split text into sentences


def fruitcrawl():
    """""Cralws Wikipedia pages we modfied fruit names to match corresponding wikipedia page."""

    fruits = ["Apple", "Banana", "Cherry", "Date_palm", "Grape", "Orange_(fruit)", "Peach", "Pear",
        "Plum", "Watermelon", "Blueberry", "Strawberry", "Mango", "Kiwifruit", "Papaya",
        "Pineapple", "Lemon", "Lime_(fruit)", "Raspberry", "Blackberry"]

    if not os.path.exists('fruit_texts'):
        os.makedirs('fruit_texts')

    for fruit in fruits:
        print(f"Crawling wiki page for {fruit}...")
        text = get_wikipedia_text(fruit)
        if text:
            with open(f'fruit_texts/{fruit}.json', 'w', encoding='utf-8') as f:
                json.dump({"fruit": fruit, "text": text}, f, ensure_ascii=False, indent=4)
        else:
            print(f"Could not retrieve text for {fruit}")


def retrieve_data(q:str):
    """""Depending on needed data by the question, this function retrieves the data from the json files."""
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


def build_similarity_matrix(sentences):
    """This function builds the similarity matrix using the cosine similarity metric."""
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences) 
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return sim_matrix


def calculate_page_rank(sim_matrix, beta=0.85, eps=1.0e-8, iterations=100000):
    """Calc. Page Rank according to class notes."""
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
    """text is the text per fruit"""
    sentences:list = preprocess_for_summary(text)
    sim_matrix = build_similarity_matrix(sentences)
    page_rank = calculate_page_rank(sim_matrix)
    summary = [sentence for sentence, rank in sorted(zip(sentences, page_rank), key=lambda x: x[1], reverse=True)[:5]]
    return summary


def summarize_text_per_fruit(fruits_texts):
    """""This function summarizes the text for each fruit and saves the summaries in a json file."""
    summaries = {}
    if not os.path.exists("fruit_summaries"):
        os.makedirs("fruit_summaries")
    
    for fruit, text in fruits_texts.items():
        summaries[fruit] = summarize_text(text)
        with open(f'fruit_summaries\\{fruit}_summary.json', 'w', encoding='utf-8') as f:
            json.dump({"fruit": fruit, "summary": summaries[fruit]}, f, ensure_ascii=False, indent=4)
    
    return summaries
    

def textsum():
    full_text:dict = retrieve_data('b')
    summarize_text_per_fruit(full_text)


def summary_list(full_summaries:dict):
    s_l = []
    for _,s in full_summaries.items():
        str_summary = " ".join(s).lower()
        s_l.append(str_summary)
    return s_l


def top_words_in_summary(n=3):
    """This function calculates the tf-idf for each fruit and takes the top n words."""
    full_summaries = retrieve_data('c')
    summary_list2 = summary_list(full_summaries)
    stop_words_set = set(stopwords.words('english'))
    top_n_words_dict = {}

    for s in summary_list2:
        words = s.split()
        words = [word for word in words if word.isalpha() and len(word) > 1]
        filtered_words = [word for word in words if word not in stop_words_set]
        unique_words = set(filtered_words)
        score = tdidf(summary_list2, unique_words)

        sorted_items = sorted(score.items(), key=lambda item: item[1], reverse=True)[:n]
        # Step 2: Select the top n items from the sorted list
        top_n_items = sorted_items[:n]
        # Step 3: Convert these items back into a dictionary
        top_n_words = dict(top_n_items)

        for word in top_n_words.keys():
            if word not in top_n_words_dict:
                top_n_words_dict[word] = score[word]

    return top_n_words_dict


def insert_tfidf_scores_to_dataframe(top_n_words_dict:dict):
    
    dataframe_copy = pd.read_csv('fruits.csv')
    for word, vector in top_n_words_dict.items():
        dataframe_copy[word] = vector
    dataframe_copy.to_csv('fruits_tdidf.csv', index=False)
 

def q8_c():
    """This function creates a dataframe with the top words as columns and the fruits as rows"""
    top_n_words_dict = top_words_in_summary()
    insert_tfidf_scores_to_dataframe(top_n_words_dict)
    return top_n_words_dict


def calculate_euclidean_distance(centeroid , data_point):
    return np.linalg.norm(np.array(data_point) - np.array(centeroid))
    

def calculate_cosine_similarity(centeroid , data_point):
    if np.linalg.norm(centeroid) != 0 and np.linalg.norm(data_point) != 0:
        return np.dot(centeroid, data_point) / (np.linalg.norm(centeroid) * np.linalg.norm(data_point))


def calculate_kmeans(feature_vectors:list, k:int, iteration:int, cosine_dis:bool = False ):
    """THE K-MEANS ALGORITHM"""

    num_of_data_points = len(feature_vectors)
    centroids = feature_vectors[:k]
    data_point_groups = np.zeros(num_of_data_points)

    for i in range(iteration):
        old_centroids = np.copy(centroids)

        for n in range(num_of_data_points):
                if cosine_dis:
                    # for section g we used cosine similarity distance
                    distances = [calculate_cosine_similarity(centroid, feature_vectors[n]) for centroid in centroids]
                    data_point_groups[n] = np.argmax(distances)

                else:
                    #for all the other sections we used the normal euclidean distance
                    distances = [calculate_euclidean_distance(centroid, feature_vectors[n]) for centroid in centroids]
                    data_point_groups[n] = np.argmin(distances)
        
        if i != iteration - 1:
            for j in range(k):
                cluster_points = [feature_vectors[index] for index in range(num_of_data_points) if data_point_groups[index] == j]
                if cluster_points:
                    centroids[j] = np.mean(cluster_points, axis=0)
 
            if np.array_equal(old_centroids, centroids):
                break



    return data_point_groups,centroids


def kmeans(graph:bool = True):
    a,b,c = "Amount of Sugar","Time it Lasts","Price"
    feature_vectors = []
    for _ , row in DataFrame.iterrows():
        feature_vectors.append(np.array([row[a], row[b], row[c]]))

    data_point_groups,centroids  = calculate_kmeans(feature_vectors,4,50)
    if graph:
        x_point = [v[0] for v in feature_vectors]
        y_point = [v[2] for v in feature_vectors]
        # centroids_x =[c[0] for c in centroids]
        # centroids_y = [c[2] for c in centroids]

        fig = make_subplots(rows=1, cols=1)
        fig.update_layout(title={'text': "Qustion 8 Section d - K-Means scored by: Amount of Sugar, Time it Lasts, Price",
                                'x': 0.5,  'y': 0.95,  'xanchor': 'center',  'yanchor': 'top'  })
        fig.update_xaxes(title_text="Amount of Sugar")
        fig.update_yaxes(title_text="Price")

        fig.add_trace(go.Scatter( x=x_point, y=y_point,name="Data Points",mode="markers", marker=dict(size=13, color=data_point_groups)))
        # fig.add_trace(go.Scatter(x=centroids_x,y=centroids_y,name="Centeroids",mode="markers",marker=dict(symbol="x", size=13)))
        fig.write_html('q8_d.html')
        fig.show()
    return feature_vectors
    

def q8_e(graph:bool = True):
    Peeling_Messiness = {'Low':0, 'Medium':0.5, 'High':1}
    seasons = {"Winter": 0,'Fall':0.25 ,"Spring": 0.75 , "Summer": 1} #valued simillarity
    colors = ['Yellow','Green','Pink', 'Orange','Red', 'Purple', 'Blue', 'Brown', 'Black'] #sorted by brightness

    a,b,c = "Peeling/Messiness", "Color", "Growth Season"

    feature_vectors = []
    for _ , row in DataFrame.iterrows():
        feature_vectors.append(np.array([Peeling_Messiness[row[a]], colors.index(row[b])/len(colors), seasons[row[c]]]))

    data_point_groups, _ = calculate_kmeans(feature_vectors,4,50)
    if graph:
        x_point = DataFrame["Amount of Sugar"].tolist()
        y_point = DataFrame["Price"].tolist()

        fig = make_subplots(rows=1, cols=1)
        fig.update_layout(title={'text': "Qustion 8 Section e - K-Means scored by:Peeling/Messiness, Color, Growth Season",
                                 'x': 0.5,  'y': 0.95,  'xanchor': 'center',  'yanchor': 'top'  })
        fig.update_xaxes(title_text="Amount of Sugar")
        fig.update_yaxes(title_text="Price")

        fig.add_trace(go.Scatter(x=x_point, y=y_point,name="Data Points",mode="markers", marker=dict(size=13, color=data_point_groups)))
        fig.write_html('q8_e.html')
        fig.show()
    return feature_vectors


def norm(v):
    norm_value = np.linalg.norm(v)
    if norm_value != 0:
        return v / norm_value
    else:
        return v 


def q8_f(top_n_words_tdidf:dict, graph:bool = True):
    feature_vectors = []
    DataFrameidf = pd.read_csv('fruits_tdidf.csv')
    
    for _ , row in DataFrameidf.iterrows():
        l = []
        for word in top_n_words_tdidf.keys():
            l.append(row[word])
        feature_vectors.append(norm(np.array(l)))
       
    data_point_groups,centroids  = calculate_kmeans(feature_vectors,4,50)
    if graph:
        x_point = DataFrame["Amount of Sugar"].tolist()
        y_point = DataFrame["Price"].tolist()

        fig = make_subplots(rows=1, cols=1)
        fig.update_layout(title={'text': "Qustion 8 Section f - K-Means, td-idf score of the summary word",
                                'x': 0.5, 'y': 0.95, 'xanchor': 'center', 'yanchor': 'top'})        
        fig.update_xaxes(title_text="Amount of Sugar")
        fig.update_yaxes(title_text="Price")

        fig.add_trace(go.Scatter(x=x_point, y=y_point,name="Data Points",mode="markers", marker=dict(size=13, color=data_point_groups)))
        fig.write_html('q8_f.html')
        fig.show()        
    return feature_vectors    


def combine_feature_vec(top_n_words_tdidf):
    v_kmeans = kmeans(graph=False)
    v_e = q8_e(graph=False)
    v_f = q8_f(top_n_words_tdidf,graph=False)

    feature_vector = []

    for i in range(len(fruit_list_with_modification)):
        v = np.concatenate((v_kmeans[i],v_e[i],norm(v_f[i])))
        feature_vector.append(v)
    return feature_vector


def q8_g(top_n_words_tdidf):

    combined_feature_vec = combine_feature_vec(top_n_words_tdidf)
    data_point_groups,centroids  = calculate_kmeans(combined_feature_vec,4,50,cosine_dis=True)
    
    x_point = DataFrame["Amount of Sugar"].tolist()
    y_point = DataFrame["Price"].tolist()

    fig = make_subplots(rows=1, cols=1)
    fig.update_layout(title={'text': "Qustion 8 Section g - K-Means\n All the Feature of Fruits",
                            'x': 0.5,  'y': 0.95,  'xanchor': 'center',  'yanchor': 'top'  })
    fig.update_xaxes(title_text="Amount of Sugar")
    fig.update_yaxes(title_text="Price")

    fig.add_trace(go.Scatter(x=x_point, y=y_point,name="Data Points",mode="markers", marker=dict(size=13, color=data_point_groups)))
    fig.write_html('q8_g.html')
    fig.show()      


if __name__ == "__main__":
    global fruit_list_with_modification 
    fruit_list_with_modification = ["Apple", "Banana", "Cherry", "Date_palm", "Grape", "Orange_(fruit)", "Peach", "Pear",
        "Plum", "Watermelon", "Blueberry", "Strawberry", "Mango", "Kiwifruit", "Papaya",
        "Pineapple", "Lemon", "Lime_(fruit)", "Raspberry", "Blackberry"]
    global DataFrame
    DataFrame = pd.read_csv('fruits.csv')
    fruitcrawl() # q_a
    textsum() #q_b
    nltk.download('stopwords')
    top_n_words_tdidf = q8_c()
    kmeans() #q_d
    q8_e()
    q8_f(top_n_words_tdidf)
    q8_g(top_n_words_tdidf)

