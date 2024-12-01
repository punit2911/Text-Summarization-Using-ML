from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import spacy_streamlit
import requests
import json
from bs4 import BeautifulSoup
import configparser
import streamlit as st
import random

# Load the spacy model
import spacy
nlp = spacy.load('en_core_web_sm')

# Set stopwords and punctuation
stopwords = list(STOP_WORDS)
punctuation = punctuation + "\n"

# Config for API key
config = configparser.ConfigParser()
config.read("config.ini")
news_api_key = config["API"]["news_api"]

def spacy_render(summary, visualize_full_article=False):
    """Render the Spacy NER visualization."""
    doc = nlp(summary)
    title = "Full Article Visualization" if visualize_full_article else "Summary Visualization"
    spacy_streamlit.visualize_ner(
        doc, labels=nlp.get_pipe("ner").labels, 
        title=title, show_table=False, 
        key=random.randint(0, 100)
    )

def word_frequency(doc):
    """Calculate word frequencies excluding stopwords and punctuation."""
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            word_frequencies[word.text] = word_frequencies.get(word.text, 0) + 1
    return word_frequencies

def sentence_score(sentence_tokens, word_frequencies):
    """Score each sentence based on word frequencies."""
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word.text.lower()]
    return sentence_scores

@st.cache_data()
def fetch_news_links(query="india"):
    """Fetch news article links, titles, and thumbnails from the API."""
    req_url = f"https://newsapi.org/v2/everything?sources=bbc-news&q={query}&language=en&apiKey={news_api_key}"
    headers = {
        "Accept": "*/*",
        "User-Agent": "Thunder Client (https://www.thunderclient.com)" 
    }
    response = requests.get(req_url, headers=headers).json()

    link_list, title_list, thumbnail_list = [], [], []
    for i, article in enumerate(response.get("articles", [])):
        if i == 10:
            break
        if "/news/" in article["url"] and "stories" not in article["url"]:
            link_list.append(article["url"])
            title_list.append(article["title"])
            thumbnail_list.append(article.get("urlToImage", ""))
    
    return link_list, title_list, thumbnail_list

@st.cache_data()
def fetch_news(link_list):
    """Fetch the text content of news articles from the provided URLs."""
    news_list = []
    headers = {
        "Accept": "*/*",
        "User-Agent": "Thunder Client (https://www.thunderclient.com)" 
    }

    for link in link_list:
        response = requests.get(link, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.findAll("div", {"data-component": "text-block"})
        article_text = " ".join([para.get_text() for para in paragraphs])
        news_list.append(article_text)
    
    return news_list

def get_summary(text, summary_length_ratio=0.10):
    """Generate a summary of the text using word frequencies."""
    doc = nlp(text)
    word_frequencies = word_frequency(doc)
    
    # Normalize word frequencies
    max_freq = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] /= max_freq
    
    sentence_tokens = list(doc.sents)
    sentence_scores = sentence_score(sentence_tokens, word_frequencies)
    
    # Select top sentences for the summary
    select_length = int(len(sentence_tokens) * summary_length_ratio)
    summary_sentences = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    
    summary = " ".join([sent.text for sent in summary_sentences])
    return summary
