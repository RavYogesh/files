import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException, Body
import uvicorn
from memory_profiler import profile
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Initialize TF-IDF vectorizer outside the function

from memory_profiler import profile
# Function to find similar utterances for a given keyword with specified minimum similarity and minimum word length


vectorizer = TfidfVectorizer () # initialize the vectorizer
 # transform the utterances into vectors


@profile
def matchk(keyword):
  df = pd.read_excel('input.xlsx')
  df = df.astype(str)
  X = vectorizer.fit_transform(df['utterance'])
  keyword_vector = vectorizer.transform ([keyword]) # transform the keyword into a vector
  scores = cosine_similarity (keyword_vector, X) # compute the similarity scores
  top_indices = np.argsort (scores) [0] [::-1] # get the indices of the matches in descending order
  matches = [] # initialize an empty list to store the matches
  seen_intents = set () # initialize an empty set to store the seen intents
  for index in top_indices: # loop through the indices
    intent = df.loc [index, 'expectedIntent'] # get the intent
    if intent not in seen_intents: # check if the intent is not already seen
      seen_intents.add (intent) # add the intent to the seen set
      utterance = df.loc [index, 'utterance'] # get the utterance
      score = scores [0] [index]
      if score > 0.6 and len(utterance.split ()) >= 3:
        matches.append ({'utterance': utterance, 'intent': intent, 'score': score}) # append a dictionary with the utterance and intent to the list
        if len (matches) == 5: # check if the list has 5 matches
            break # stop the loop
  return matches # return the list of matches

# FastAPI endpoint to find matching utterances for a keyword
@app.post("/match/")
async def match_keyword(keyword: str = Body(..., embed=True)):
    try:
        # Read the Excel file

        matching_utterances = matchk(keyword)
        return {"matching_utterances": matching_utterances}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
   uvicorn.run(app, port=8006)
