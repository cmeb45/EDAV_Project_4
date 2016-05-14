import pandas as pd
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
from nltk import tokenize
import numpy as np
import matplotlib.pyplot as plt

# MANUALLY FIX LINES IN THIS FILE SO THAT THERE ARE NO EXTRA SPACES
# IN BETWEEN LINES OF PRESIDENTS & DATES OF SPEECHES

# MANUALLY REMOVE EXTRA LINES IN BETWEEN SPEECHES

f = open('State+of+the+Union+Addresses+1970-2016.txt')

lines = f.readlines()
bigline = " ".join(lines)
stars = bigline.split('***')
splits = [s.split('\n') for s in stars[1:]]
tups = [(s[2].strip(), s[3].strip(), s[4].strip(), "".join(s[5:])) for s in splits]
# ("State of the Union Address", President, Date of Speech, Speech)
speech_df = pd.DataFrame(tups)


# AVERAGE SENTIMENT SCORES OF PRESIDENTS ACROSS ALL SPEECHES
sentence_dict_pres = {}
sentence_dict_pres["Richard Nixon"] = []
sentence_dict_pres["Gerald R. Ford"] = []
sentence_dict_pres["Jimmy Carter"] = []
sentence_dict_pres["Ronald Reagan"] = []
sentence_dict_pres["George H.W. Bush"] = []
sentence_dict_pres["William J. Clinton"] = []
sentence_dict_pres["George W. Bush"] = []
sentence_dict_pres["Barack Obama"] = []


for i in range(len(speech_df)):
    if speech_df[1][i] in ["Richard Nixon","Gerald R. Ford",
    "Jimmy Carter","Ronald Reagan","George H.W. Bush",
    "William J. Clinton","George W. Bush","Barack Obama"]:
        lines_list = tokenize.sent_tokenize(speech_df[3][i])
        sentence_dict_pres[speech_df[1][i]].extend(lines_list)

sentiment_dict_pres = {}
sentiment_dict_pres["Richard Nixon"] = 0.0
sentiment_dict_pres["Gerald R. Ford"] = 0.0
sentiment_dict_pres["Jimmy Carter"] = 0.0
sentiment_dict_pres["Ronald Reagan"] = 0.0
sentiment_dict_pres["George H.W. Bush"] = 0.0
sentiment_dict_pres["William J. Clinton"] = 0.0
sentiment_dict_pres["George W. Bush"] = 0.0
sentiment_dict_pres["Barack Obama"] = 0.0

for president in ["Richard Nixon","Gerald R. Ford",
    "Jimmy Carter","Ronald Reagan","George H.W. Bush",
    "William J. Clinton","George W. Bush","Barack Obama"]:
    temp_score = []
    for sentence in sentence_dict_pres[president]:
        vs = vaderSentiment(sentence)
        temp_score.append(vs['compound'])
    sentiment_dict_pres[president] = np.mean(temp_score)



# AVERAGE SENTIMENT SCORE OF SENTENCES IN EACH SPEECH
sentence_dict_speech = {}
for i in range(len(speech_df)):
    if speech_df[1][i] in ["Richard Nixon","Gerald R. Ford",
    "Jimmy Carter","Ronald Reagan","George H.W. Bush",
    "William J. Clinton","George W. Bush","Barack Obama"]:
        sentence_dict_speech[speech_df[2][i]] = []


for i in range(len(speech_df)):
    if speech_df[2][i] in sentence_dict_speech.keys():
        lines_list = tokenize.sent_tokenize(speech_df[3][i])
        sentence_dict_speech[speech_df[2][i]].extend(lines_list)

sentiment_dict_speech = {}
for day in sentence_dict_speech.keys():
    temp_score = []
    for sentence in sentence_dict_speech[day]:
        vs = vaderSentiment(sentence)
        temp_score.append(vs['compound'])
    sentiment_dict_speech[day] = np.mean(temp_score)
