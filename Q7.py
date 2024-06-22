import pandas as pd
import numpy as np
import pprint
import matplotlib.pyplot as plt
from pandas.plotting import table


def td(descriptions, words:list):
    td_words = {}
    for word in words:
        td_word = []
        for doc in descriptions:
            td_word.append(doc.count(word))
        td_words[word] = td_word
    return td_words


def in_how_many_docs_appears(description, word:str):
    c=0
    for doc in description:
        if word in doc:
            c+=1
    return c
    
def idf(num_of_docs:int, descriptions ,words:list):
    idf_words = {}
    for word in words:
        word_count = in_how_many_docs_appears(descriptions, word)
        if word_count != 0:
            idf_words[word] = np.log10(num_of_docs/word_count)
        else:
            idf_words[word] = 0
    return idf_words


def compute_tdidf(tds, idfs):
    tfidf_words = {}
    for word in idfs:
        l = []
        for n in tds[word]:
            l.append(n*idfs[word])
        tfidf_words[word] = l
    return tfidf_words


def tdidf(descriptions,words):
    tds = td(descriptions, words)
    idfs = idf(len(descriptions), descriptions,words)
    tfidf_values = compute_tdidf(tds, idfs)
    return tfidf_values



def save_dataframe_as_image(df, filename):
    fig, ax = plt.subplots(figsize=(12, 8))  
    ax.axis('tight')
    ax.axis('off')
    
    df_rounded = df.round(6)
    
    tbl = table(ax, df_rounded, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)
    plt.savefig(filename, bbox_inches='tight', dpi=300)


def q7():
    df = pd.read_csv('music_festivals.csv')
    descriptions = df['Description'].str.lower().tolist()
    words = ['annual', 'music', 'festival', 'soul', 'jazz', 'belgium', 'hungary', 'israel',
              'rock', 'dance', 'desert', 'electronic', 'arts']
    values = tdidf(descriptions,words)

    tfidf_df = pd.DataFrame(values)
    tfidf_df['Music Festival'] = df['Music Festival']
    tfidf_df.set_index('Music Festival', inplace=True)
        
    save_dataframe_as_image(tfidf_df, 'tfidf_table.png')


if __name__ == "__main__":
    q7()