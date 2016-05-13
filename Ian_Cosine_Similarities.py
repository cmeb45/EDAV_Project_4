
# coding: utf-8

# In[1]:

import numpy as np
import numpy.linalg as LA
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# In[2]:

f = open('State of the Union Addresses 1970-2016.txt')
lines = f.readlines()
bigline = " ".join(lines)
stars = bigline.split('***')
splits = [s.split('\r\n') for s in stars[1:]]
tups = [(s[2], s[3], s[4], "".join(s[5:])) for s in splits]
df = pd.DataFrame(tups)


# In[3]:

df.columns = ['title', 'pres', 'date', 'speech']
df['year'] = df['date'].str.split(',', expand=True)[1]


# In[4]:

tfidf_orig = TfidfVectorizer(stop_words='english')
tfidf = tfidf_orig.fit_transform(df['speech'])
cosine_similarities = linear_kernel(tfidf, tfidf)

# Most highly-related speeches
related_docs_indices = cosine_similarities.argsort()[:-5:-4]
# print(related_docs_indices)

cosine_similarities = pd.DataFrame(cosine_similarities)

# Set Column and Row names for graphing
cosine_similarities.columns = df['pres'] + df['year']
cosine_similarities = cosine_similarities.set_index(df['pres'] + df['year'])


# In[17]:

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(9, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(250, 10, n=3, as_cmap=True)

h = sns.heatmap(cosine_similarities.iloc[178:, 178:], #mask=mask, cmap=cmap, #vmax=.3,
            square=True, #xticklabels=5, yticklabels=5,
            linewidths=0, cbar_kws={"shrink": .5}, ax=ax)

plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

