import pandas as pd
df = pd.read_excel ('utterances.xlsx') # assuming your file is named 'utterances.xlsx'

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
vectorizer = TfidfVectorizer () # initialize the vectorizer
X = vectorizer.fit_transform (df['utterance']) # transform the utterances into vectors


import numpy as np
def match_keyword (keyword):
  keyword_vector = vectorizer.transform ([keyword]) # transform the keyword into a vector
  scores = cosine_similarity (keyword_vector, X) # compute the similarity scores
  top_indices = np.argsort (scores) [0] [::-1] # get the indices of the matches in descending order
  matches = [] # initialize an empty list to store the matches
  seen_intents = set () # initialize an empty set to store the seen intents
  for index in top_indices: # loop through the indices
    intent = df.loc [index, 'intent'] # get the intent
    if intent not in seen_intents: # check if the intent is not already seen
      seen_intents.add (intent) # add the intent to the seen set
      utterance = df.loc [index, 'utterance'] # get the utterance
      matches.append ({'utterance': utterance, 'intent': intent}) # append a dictionary with the utterance and intent to the list
      if len (matches) == 5: # check if the list has 5 matches
        break # stop the loop
  return matches # return the list of matches

from flask import Flask, request, jsonify
app = Flask (__name__) # create the app object
@app.route ('/match', methods= ['POST']) # define the endpoint URL and method
def match ():
  keyword = request.json ['keyword'] # get the keyword from the request json
  matches = match_keyword (keyword) # call the match_keyword function
  return jsonify (matches) # return the matches as a response json

if __name__ == '__main__':
  app.run (debug=True) # run the app in debug mode
