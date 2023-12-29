## mhttps://github.com/jonaschn/awesome-topic-models

A subfield of natural language can reveal the underlying structure in large amounts of text. This discipline is called Topic Modeling, that is specialized in extracting topics from text.

In this context, conventional approaches, like Latent Dirichlet Allocation and Non-Negative Matrix Factorization, demonstrated to not capture well the relationships between words since they are based on bag-of-word.

For this reason, we are going to focus on two promising approaches, Top2Vec and BERTopic, that address these drawbacks by exploiting pre-trained language models to generate topics. Let’s get started!

# Topic Modeling Approaches: Top2Vec vs BERTopic

# Top2Vec

Top2Vec is a model capable of detecting automatically topics from the text by using pre-trained word vectors and creating meaningful embedded topics, documents and word vectors.

In this approach, the procedure to extract topics can be split into different steps:

Create Semantic Embedding: jointly embedded document and word vectors are created. The idea is that similar documents should be closer in the embedding space, while dissimilar documents should be distant between them.
Reduce the dimensionality of the document embedding: The application of the dimensionality reduction approach is important to preserve most of the variability of the embedding of documents while reducing the high dimensional space. Moreover, it allows to identification of dense areas, in which each point represents a document vector. UMAP is the typical dimensionality reduction approach chosen in this step because it’s able to preserve the local and global structure of the high-dimensional data.
Identify clusters of documents: HDBScan, a density-based clustering approach, is applied to find dense areas of similar documents. Each document is assigned as noise if it’s not in a dense cluster, or a label if it belongs to a dense area.
Calculate centroids in the original embedding space: The centroid is computed by considering the high dimensional space, instead of the reduced embedding space. The classic strategy consists in calculating the arithmetic mean of all the document vectors belonging to a dense area, obtained in the previous step with HDBSCAN. In this way, a topic vector is generated for each cluster.
Find words for each topic vector: the nearest word vectors to the document vector are semantically the most representative.


# BERTopic

### "BERTopic is a topic modeling technique that leverages transformers and c-TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions."

As the name suggests, BERTopic utilises powerful transformer models to identify the topics present in the text. Another characteristic of this topic modeling algorithm is the use of a variant of TF-IDF, called class-based variation of TF-IDF.

Like Top2Vec, it doesn’t need to know the number of topics, but it automatically extracts the topics.

Moreover, similarly to Top2Vec, it is an algorithm that involves different phases. The first three steps are the same: creation of embedding documents, dimensionality reduction with UMAP and clustering with HDBScan.

The successive phases begin to diverge from Top2Vec. After finding the dense areas with HDBSCAN, each topic is tokenized into a bag-of-words representation, which takes into account if the word appears in the document or not. After the documents belonging to a cluster are considered a unique document and TF-IDF is applied. So, for each topic, we identify the most relevant words, that should have the highest c-TF-IDF.


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 

# Existing Word Embedding Techniques are - 
1. BERT
2. FinBert
3. RoBerta
4. DistillBert

# LDA & LSA
Matrix based models.
Topics are produced based on word frequency.
Fail to work on large corpus as they have sparsity issue.
Ignores the order of words, hence semantic relationship between the words is not captured.


# Embedded Space Models
Vector based models.
Topics are extracted by grouping the similar meaning words together
Works effectively on large corpus.
Considers the order of words and captures the semantic relationship between the words.
