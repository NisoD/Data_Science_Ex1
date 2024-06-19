import json
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix
from scipy.linalg import eig
import os
import numpy as np

def tf_idf(text):
    """Calculates TF-IDF scores for words in a text."""
    word_counts = defaultdict(int)
    document_count = 0

    for sentence in text:
        document_count += 1
        for word in word_tokenize(sentence.lower()):
            if word not in stopwords.words('english'):  # Remove stopwords
                word_counts[word] += 1

    tf_idfs = {}
    for word, count in word_counts.items():
        df = sum(word in sentence for sentence in text)  # Document frequency
        tf = count / len(word_tokenize(' '.join(text)))  # Term frequency
        idf = np.log((document_count + 1) / (df + 1))  # Inverse document frequency
        tf_idfs[word] = tf * idf

    return tf_idfs

def similarity_matrix(text, tf_idfs):
    """Constructs a similarity matrix based on TF-IDF scores."""
    num_sentences = len(text)
    similarity_matrix = csr_matrix((num_sentences, num_sentences), dtype=float)

    for i, sentence1 in enumerate(text):
        for j, sentence2 in enumerate(text):
            if i == j:
                continue  # Avoid self-similarity

            sentence1_words = set(word_tokenize(sentence1))
            sentence2_words = set(word_tokenize(sentence2))
            similarity = sum(tf_idfs.get(word, 0) for word in sentence1_words.intersection(sentence2_words))
            similarity_matrix[i, j] = similarity

    return similarity_matrix

def pagerank(similarity_matrix, damping_factor=0.85):
    """Implements the PageRank algorithm for sentence ranking."""
    A = similarity_matrix.toarray()
    n = A.shape[0]

    # Handle dangling nodes by creating a uniform transition matrix
    dangling_sums = np.where(A.sum(axis=1) == 0, 1 / n, 0)
    transition_matrix = (damping_factor * A) + ((1 - damping_factor) * dangling_sums[:, np.newaxis])

    # Calculate PageRank scores using power method
    eigenvalues, eigenvectors = eig(transition_matrix.T)
    pagerank_scores = eigenvectors[:, 0].real  # Select the first eigenvector (dominant)
    pagerank_scores /= pagerank_scores.sum()  # Normalize scores

    return pagerank_scores

def textsum(json_files, num_sentences=5):
    """Summarizes text information from JSON files using PageRank."""
    summaries = {}
    for filename in json_files:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            text = data.get('description', '')

        sentences = text.split('.')  # Split into sentences (adjust as needed)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]  # Remove empty sentences

        tf_idfs = tf_idf(sentences)
        sim_matrix = similarity_matrix(sentences, tf_idfs)
        pagerank_scores = pagerank(sim_matrix)

        top_sentence_indices = np.argsort(pagerank_scores)[::-1][:num_sentences]  # Sort and select top indices
        summary = '. '.join([sentences[i] for i in top_sentence_indices])
        summaries[data.get('name', 'Unknown')] = summary

    return summaries

# Example usage (assuming you have your JSON files)
json_files = [os.path.join('fruit_texts', file) for file in os.listdir('fruit_texts') if file.endswith('.json')]
summaries = textsum(json_files)

for fruit, summary in summaries.items():
    print(f"{fruit} summary:")
    print(summary)
    print()
