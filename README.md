# Graph-Embeddings-for-Recommender-Systems

**Overview**

In this project, we will revisit the problem central to recommender systems: predicting a user’s
preference for some item they have not yet rated. Like the Spark recommender from the first
project, we will use a collaborative filtering model to explore this problem. Recall that in this
model, the goal is to find the sentiment of a user about a particular item Unlike the the first
project that used the ALS method, however, we will perform this task using a graphbased
technique called DeepWalk. The main steps that you will complete are to:

1. Create a heterogeneous information network with nodes consisting of users, itemratings, items, and other entities related to those items
2. Use DeepWalk to generate random walks over this graph
3. Based on these random walks, embed the graph in a low dimensional space using word2vec.

The goal of this project is to evaluate and compare p reference propagation algorithms in
heterogeneous information networks generated from useritem relationships. Specifically, you will implement and evaluate a word2vecbased method.

**Background**

Useritem relationships can be represented as heterogeneous information network. In such a
network, entities of different categories are placed as nodes in the same network (e.g., users,
businesses, actors, directors). The edges between nodes represent some form of a relationship.
The networkbased
entity recommendation systems, which utilize useritem
relationship
information, require a method for preference propagation over the network [1].

Data embedding is used in machine learning applications to create low dimensional
feature
representations, which preserves the structure of data points in their original space. Embedding
means low dimensional vector representation of entities. In particular, information network
embedding is representation of a network in low dimensional vector space which preserves
information about the structure of the network. The main goal of the heterogeneous embedding
task is to learn mapping functions to project data from different modalities to a common space
so that similarities between objects can be directly measured.

One of the recent efficient to obtain a network embedding is DeepWalk [4]. DeepWalk produces
the embeddings of nodes of the information network by first creating corpus of text, where
words are node ids and sentences are random walks originated at each node of predefined
length. After generation of the randomwalk
corpus, DeepWalk feeds it to word2vec, which
produces the embeddings. word2vec, a shallow neural networkbased
set of algorithms [2] has
been applied outside of the natural language processing (NLP) domain to, effectively map latent
features hidden in the paths over the information network into a vector space.

It’s been shown that the resulting low dimensional embeddings, which are produced by
word2vec, are equivalent to factors in matrix factorization approaches [3] and [5]. Note, that in
this project, word2vec has not been used for its originally intended NLP purpose as, for
example, for extraction of sentiment from user reviews. Rather, word2vec, will be used as a
lowdimensional
embedding tool, capitalizing on the fact that it is agnostic to the type of
vocabulary and corpus that is fed to it.

**Running the program**

The program can be run using the following command from root folder (not inside rec2vec):

* python -m rec2vec --walk-length 2 --number-walks 2 --workers 4
