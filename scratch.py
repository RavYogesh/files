Sure, I can modify the match_keyword function to output the similarity score of utterances as well. To do this, you will need to add the score to the dictionary that is appended to the matches list. For example:

Python

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
      score = scores [0] [index] # get the score
      matches.append ({'utterance': utterance, 'intent': intent, 'score': score}) # append a dictionary with the utterance, intent, and score to the list
      if len (matches) == 5: # check if the list has 5 matches
        break # stop the loop
  return matches # return the list of matches
AI-generated code. Review and use carefully. More info on FAQ.
This function will return a list of dictionaries that contain the utterance, intent, and score for each match. The score is a value between 0 and 1, where 1 means the keyword and the utterance are identical, and 0 means they are completely different. The score is computed using the cosine similarity, which measures the angle between two vectors1. You can learn more about cosine similarity and how to compute it in Python from this web page.

I hope this helps you with your task. If you have any other questions, please feel free to ask. 😊

1: Cosine similarity - Wikipedia 2: How to compute the similarity between two text documents? - Stack Overflow