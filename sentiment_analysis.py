import pandas as pd
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
from nltk import tokenize
import numpy as np

# MANUALLY FIX LINES IN THIS FILE SO THAT THERE ARE NO EXTRA SPACES
# IN BETWEEN LINES OF PRESIDENTS & DATES OF SPEECHES

# MANUALLY REMOVE EXTRA LINES IN BETWEEN SPEECHES

f = open('State+of+the+Union+Addresses+1970-2016.txt')

lines = f.readlines()
bigline = " ".join(lines)
stars = bigline.split('***')
splits = [s.split('\n') for s in stars[1:]]
tups = [(s[2].strip(), s[3].strip(), s[4].strip(), "".join(s[5:])) for s in splits]
speech_df = pd.DataFrame(tups)


sentence_dict = {}
sentence_dict["Richard Nixon"] = []
sentence_dict["Gerald R. Ford"] = []
sentence_dict["Jimmy Carter"] = []
sentence_dict["Ronald Reagan"] = []
sentence_dict["George H.W. Bush"] = []
sentence_dict["William J. Clinton"] = []
sentence_dict["George W. Bush"] = []
sentence_dict["Barack Obama"] = []


for i in range(len(speech_df)):
    if speech_df[1][i] in ["Richard Nixon","Gerald R. Ford",
    "Jimmy Carter","Ronald Reagan","George H.W. Bush",
    "William J. Clinton","George W. Bush","Barack Obama"]:
        lines_list = tokenize.sent_tokenize(speech_df[3][i])
        sentence_dict[speech_df[1][i]].extend(lines_list)

sentiment_nixon = []
for sentence in sentences_nixon:
    vs = vaderSentiment(sentence)
    sentiment_nixon.append(vs['compound'])

np.mean(sentiment_nixon)
# 0.26135437956204383
