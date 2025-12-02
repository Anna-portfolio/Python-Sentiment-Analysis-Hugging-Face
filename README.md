# Python Sentiment Analysis – Hugging Face Dataset
created by Anna Dudek @Anna-portfolio

## Overview

This project implements an end-to-end sentiment analysis pipeline using classic NLP techniques.  
It builds a Naive Bayes classifier to detect positive or negative sentiment in movie reviews sourced from the Hugging Face Rotten Tomatoes dataset.  
<br>

The workflow demonstrates:

* Text preprocessing with spaCy (cleaning, tokenization, lemmatization, stopword filtering)  
* Vectorization using CountVectorizer
* Model training with Multinomial Naive Bayes
* Evaluation on validation data  
* Visualization of dataset class distribution  
* Example prediction on custom input text  

The project provides an interpretable baseline sentiment model using well-established NLP methods, making it easy to extend or compare against transformer-based approaches.  
<br>

## Dataset

* **Source:** Hugging Face – Rotten Tomatoes Reviews  
* **Link:** https://huggingface.co/datasets/stanfordnlp/imdb  
* Contains short movie review excerpts labeled as:
  * `1` – positive  
  * `0` – negative  
* Total samples: 8,530

The dataset is loaded using the `datasets` library and converted into a Pandas DataFrame for preprocessing and analysis.  
<br>

## Project Workflow

### 1. Data Loading & Validation

* Load the Rotten Tomatoes dataset via Hugging Face `datasets`
* Convert to Pandas DataFrame
* Validate data quality (if missing values or duplicates exist)
* Inspect structure and class distribution

### 2. Text Cleaning

A custom `clean_text()` function prepares raw text by:
* converting to lowercase  
* removing punctuation/special characters (regex)  
* normalizing spacing  

### 3. Tokenization & Lemmatization (spaCy)

Using English spaCy pipeline:
* tokenize text  
* apply lemmatization  
* remove stopwords  
* return clean token list for vectorization  

### 4. Vectorization

The project uses:

* CountVectorizer (Bag-of-Words model)  
* Custom tokenizer for spaCy integration  
* Produces a sparse vector matrix:  

