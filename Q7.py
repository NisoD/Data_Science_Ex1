import pandas as pd
import numpy as np
import pprint
import matplotlib.pyplot as plt
from pandas.plotting import table

def tf(documents, words):
        """returns a dictionary with words keys and list of all tf per document as values"""
        return {word: [doc.split().count(word) / len(doc.split()) for doc in documents] for word in words}

def idf(num_of_docs, tf_words):
        return {word: np.log(num_of_docs / (1 + sum(1 for tf in tf_list if tf > 0))) for word, tf_list in tf_words.items()}

    

def compute_tfidf(tfs, idfs):
    tfidf_words = {}
    for word in tfs:
        tfidf_word = [tf * idfs[word] for tf in tfs[word]]
        tfidf_words[word] = tfidf_word
    return tfidf_words

def save_dataframe_as_image(df, filename):
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the size as needed
    ax.axis('tight')
    ax.axis('off')
    
    # Round the DataFrame to 4 decimal places
    df_rounded = df.round(4)
    
    tbl = table(ax, df_rounded, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)
    plt.savefig(filename, bbox_inches='tight', dpi=300)

def main():
    file = 'music_festivals.csv'
    df = pd.read_csv(file)
    descriptions = df['Description'].str.lower().tolist()
    words = ['annual', 'music', 'festival', 'soul', 'jazz', 'belgium', 'hungary', 'israel', 'rock', 'dance', 'desert', 'electronic', 'arts']
    
    tfs = tf(descriptions, words)
    idfs = idf(len(descriptions), tfs)
    tfidf_values = compute_tfidf(tfs, idfs)
    
    # Convert to DataFrame for better readability
    tfidf_df = pd.DataFrame(tfidf_values)
    tfidf_df['Music Festival'] = df['Music Festival']
    tfidf_df.set_index('Music Festival', inplace=True)
    
    pprint.pprint(tfidf_df)
    
    # Save the DataFrame as an image
    save_dataframe_as_image(tfidf_df, 'tfidf_table.png')
    
main()
