To accomplish this task, you can use the following Python code. Make sure to replace `'your_excel_file.xlsx'` with the path to your Excel file:

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

# Read the Excel file
df = pd.read_excel('your_excel_file.xlsx')

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['utterance'])

# Function to find top 5 similar utterances for a given keyword, each belonging to a unique intent
def find_top_similar(keyword):
    keyword_vector = tfidf_vectorizer.transform([keyword])
    cosine_similarities = cosine_similarity(keyword_vector, tfidf_matrix)
    
    # Get indices of top 5 matches
    top_indices = cosine_similarities.argsort()[0][-5:][::-1]
    
    # Initialize dictionary to store matching utterances with corresponding intents
    matching_utterances = {}
    
    # Iterate over top indices and extract corresponding utterances and intents
    for idx in top_indices:
        intent = df['intent'][idx]
        utterance = df['utterance'][idx]
        
        # If intent already exists in dictionary, append utterance; otherwise, create new key
        if intent in matching_utterances:
            matching_utterances[intent].append(utterance)
        else:
            matching_utterances[intent] = [utterance]
    
    return matching_utterances

# Set up Flask server
app = Flask(__name__)

@app.route('/find_matches', methods=['POST'])
def find_matches_handler():
    request_data = request.json
    keyword = request_data.get('keyword', '')
    matching_utterances = find_top_similar(keyword)
    return jsonify(matching_utterances)

if __name__ == '__main__':
    app.run(debug=True)
```

This code defines a Flask app with a single endpoint `/find_matches` that accepts POST requests with JSON data containing the keyword to be matched. It then returns a JSON response containing the top 5 matching utterances for each unique intent in the Excel file.