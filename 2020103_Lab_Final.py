#SENTIMENT ANALYSIS

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../2020103_Lab_Final"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
tweets=pd.read_csv("chatgpt1.csv",encoding = "ISO-8859-1")
tweets.head(50000)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *

from nltk import tokenize

sid = SentimentIntensityAnalyzer()

tweets['sentiment_compound_polarity']=tweets.Text.apply(lambda x:sid.polarity_scores(x)['compound'])
tweets['sentiment_neutral']=tweets.Text.apply(lambda x:sid.polarity_scores(x)['neu'])
tweets['sentiment_negative']=tweets.Text.apply(lambda x:sid.polarity_scores(x)['neg'])
tweets['sentiment_pos']=tweets.Text.apply(lambda x:sid.polarity_scores(x)['pos'])
tweets['sentiment_type']=''
tweets.loc[tweets.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'
tweets.loc[tweets.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'
tweets.loc[tweets.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'
tweets.head(50000)


tweets.sentiment_type.value_counts().plot(kind='bar',title="sentiment analysis")


#USER CLASSIFICATION


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np


df=pd.read_excel('chatgpt1.xlsx')

# Split the dataset into a training set and a testing set
X=df.iloc[:,:-1]
y= df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Define the feature set and target variable
df['Tweet Id'] = df['Tweet Id'].astype(str)
df['Username'] = df['Username'].astype(str)
X = df['Tweet Id']
y = df['Username']


# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train a classifier (e.g., KNN) on the training set
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Evaluate the classifier on the testing set
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))


#ENGAGEMENT PREDICTION

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('chatgpt1.csv')


# Define the feature set and target variables
X = df['Text']
y_retweets = df['RetweetCount']
y_likes = df['LikeCount']
y_replies = df['ReplyCount']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_retweets, test_size=0.2, random_state=42)


# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)



# Train a Random Forest model
rf = RandomForestRegressor()
rf.fit(X_train_tfidf, y_train)


# Predict the engagement metrics for the test set
y_pred = rf.predict(X_test_tfidf)

# Evaluate the model performance using mean squared error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean squared error:', mse)
print('R-squared:', r2)



#Hashtag analysis



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import spacy
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from string import punctuation
import collections
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import en_core_web_sm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('chatgpt1.xlsx')


from sklearn.cluster import KMeans



# Read the XLSX file
data = pd.read_csv('chatgpt1.csv')

# Extract hashtags from the 'hashtag' column
hashtags = data['hashtag'].dropna().tolist()

# Count the frequency of each hashtag
hashtag_counts = {}
for hashtag in hashtags:
    if isinstance(hashtag, str):
        if hashtag in hashtag_counts:
            hashtag_counts[hashtag] += 1
        else:
            hashtag_counts[hashtag] = 1

# Sort the hashtags by frequency in descending order
sorted_hashtags = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)

# Get the top 10 hashtags and their frequencies
top_hashtags = sorted_hashtags[:10]
hashtags, counts = zip(*top_hashtags)

# Create a bar graph for the top hashtags
plt.bar(hashtags, counts)
plt.xlabel('Hashtags')
plt.ylabel('Frequency')
plt.title('Top 10 Hashtags')
plt.xticks(rotation=45)

# Display the graph
plt.tight_layout()
plt.show()



#Clustering

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_excel('chatgpt1.xlsx')


#CLUSTERING
# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(df['Tweet Id'].astype(str))

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, n_init=10)  # Explicitly set n_init to suppress the warning
kmeans.fit(X_vectorized)

#k = 5  # Number of clusters
#kmeans = KMeans(n_clusters=k)
#kmeans.fit(X_vectorized)

# Evaluate the clustering algorithm
silhouette_avg = silhouette_score(X_vectorized, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)

# Plot the elbow curve to choose the optimal number of clusters
k_values = range(2, 10)
inertia_values = []
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_vectorized)
    inertia_values.append(kmeans.inertia_)
plt.plot(k_values, inertia_values)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Curve')
plt.show()
