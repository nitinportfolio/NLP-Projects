# Natural Language Processing

## Introduction
1. Understanding - Phonemes, Morphemes, Lexemes, Context, Syntax, Grammer
2. Strings & Reg Expressions
3. ML Models - Naive Bayes, SVM, Ensemble, Deeplearning
4. Deep Learning - RNN, LSTM, GRU, CNN, 
5. Seq2Seq Model 
6. Attention Mechanisms -  Transformers, BERT, GPT  

## Data 
1. Text FIles
2. PDFs
3. APIs
4. Web Scrapping
5. Database


## Datasets
1. Kaggle
2. paperswithcode.com/dataset
3. Quantumstat.com
4. Twitter

## Deep Learning NLP Architechtures
1. RNN
2. LSTM
3. GRU/CNN
4. Transformers
5. Autoencoders
6. To be continued

## Challenges in NLP
1. Ambiguity 
    a. I saw the man with the telescope
    b. The car hit the pole while it was moving
2. Contextual Words
    a. I ran to the store because we ran out of food
3. Colloquislisms & Slang
    a. Hit the road
    b. FOMO
    c. YOLO
    d. Dude
    e. Yep
    f. No worries
4. Synonymns
5. Iron, Sarcasm, & tonal diff
6. Spelling Error
7. Creativity
    a. Poems
    b. Dialogue
    c. Scripts
8. Diversity - so many different languages



## Libraries 
1. NLTK
2. Spacy
3. Gensim
4. TextBlob
5. Hugging Face Transformers

## Text Preprocessing
1. Removing Punchations, Stopwords
2. Tokenization
3. Stemming & Lemmatization
4. POS Tagginh
5. NER Tagging
6. Parsing
7. Coreference Resolution

## TExt Analytics & Mining

## TExt Representation
1. One Hot Encoding
2. Bag of Words
3. Tf-Idf
4. N-Grams, Unigrams, Bi-Grams
5. Word Embeddings - Word2Vec, Doc2Vec, Glove, ELMO etc

## Important Topics
1. LDA
2. SVD
3. NMF
4. Knowledge Distilation - Transfering knowledge from large model to a smaller one.


## Basic Applications
1. Text/Document Classification
2. NER - Named Entity Recognition
3. Sentiment Analysis
4. Topic Modelling 
5. TExt Clustering
6. Transfer Learning
7. Machine Translation
8. Q&A System
9. Text Summarization
10. Chatbot
11. Speech Recognition
12. Information Extraction
13. Informatio REtrieval
14. Paraphrasing
15. Conversational Agents
16. Knowledge Graph & Ques Ans system
17. TExt Generation
18. Speech to Text
19. Text to Speech

## Open Source 
1. Hugging face
2. Open AI

## Real World Applications
1. Contextual Advertisements
2. Email Clients - Spam filtering, smart reply, auto complete, much more
3. Social Media - removing content, opition mining
4. Search Engines
5. ChatBot
6. To be continued

ML vs DL in NLP
In ML we convert text to numbers, using this we loose the sequencetial information of text. In deep learning we retain & remember the sequenctial information. 
In ML we have to generate the features but in DL model creates the features itself.

# NLP Pipeline
### NLP Pipeline is a set of steps followed to build an end to end NLP software. 
- It consists of below steps but can be different based on applications
1. Data Aquisition
2. Text Preprocessing
3. Feature Engineering
4. Modeling & Evaluation
5. Deployment, Monitoring & Model Update

## NLP Projects
1. Social Listening
2. 


# Working on Text data
1. Data Acquisition
2. Data Preprocessing
    a. Cleaning
        1. html/tag 
        2. emoji
        3. spelling check
    b. Basic 
        1. Tokenization - Word/ Sentence
    c. Optional
        1. stop word removal
        2. stemming/ Lemitization
        3. removing digits, punchuations
        4. Lower casing
        5. Language detection
3. Advanced Preprocessing
    a. POS tagging
    b. Parsing
    c. Corerrerence resolution
4. Feature Engineering - Text to numbers/Vector
    a. Bag of Words
    b. TFIDF
    c. OHE
    d. Word2Vec/Doc2Vec - Deep learning
    - ML pipeline will have different feature engineering compared to DL
    - ML - We have to have a domain knowledge to create our own features. Its interpratible
    - DL - We dont have to create the fetures, DL model creates featues for us, hence domain knowledge is not required. But model is not interpretable
5. Model Building 
    a. ML Algo
    b. DL Algo - Transfer Learning like BERT etc
    c. Cloud API - OPenAI, ChatGPT API, HuggingfaceAPI, Azure API (out of box frame work) etc
    - Based on amount of data & Nature of problem we select model
6. Evaluation
    a. Intrinsic Evaluation - Use metrics to find out accuracy, perplexity
    b. Extrinsic Evaluation - After deployment (when user started using it. How often user using the feature) how is the performance
7. Deployment
    a. API  - Microservice
    b. Chatbot
8. Monitoring
    a. Dashboard with Metrices
9. Update
    a. Update with time based on change in data
    b. Online live update as in new data comes
    c. etc

# ###########################################################################################
#                             Text-Representation-BagOfWords-TFIDF-NGrams


## Feature Exraction - Text to Numbers in NLP
- Also called Text Represntation or Text Vectorization
- Vectorization should be in such a way thay it tells  the hidden meaning - semantic meaning

## Techniques for Vectorization
1. One Hot Encoding
2. Bag of Words
3. ngrams
4. TFIDF
5. Custom Features

## Techniques for Vectorization
1. One Hot Encoding - Not used - Disadvantage - Sparsity & overfitting, No Fixed Size, OOV (out of vocabulary problem), no capturing sematic meaning
2. Bag of Words - Text classification perform good - Based on Frequency of words - DisAdv - Sparsity, OOV, Ordering of word (meaning changes), Not able to capture small changes like (i am going & I am not going) both are very different meaning.
3. ngrams - Able to capture semantic better if we take value 2 or 3. DisAdv - Slow, more sparsity, 
4. TFIDF - Term frequency & Inverse Doc frequency. Get the wt & multiplied together
5. Custom Features
6. Word2Vec - Embedding - Deep learning (Best)

# Say that we have dataset of 5000 reviews
- Corpus - All the text - reviews combined together or dataset joined together
- Vocabulary - All unique words that  you have in a corpus
- Document - All the individual reviews are individual documents
- Word - simple individual words

# Tf-TDF
Term Frequency(t,d) = freq. of term t in document d **/ total number of terms in document d**
- TF - acts like a probability

Inverse Document Frequency = **log**(Total no. of documents in the corpus **/ No. of documents with term t in them**)
- IDF - Gives less wt to the common words
- We take log so that value of idf is comparable to value of tf. Reducing hte value of idf by taking the log
- Used in search engine to get the key word retrival
- Sparsity, OOV, Dimension, No semantic, not capturing the meaning

# Word Embeddings
- Converting words into real value vector that encodes the meaning of the word such that the words that are closer in the vector speace are expected to be similar in meaning.

# Types 
1. **Frequency** 
-    a. Bag of Words
-    b. tfidf
-   c. Glove - Global Vector
-   d. Matrix Factorization
2. **Prediction usign ML/DL model**
-    a. Word2Vec - Deep Learing Based
