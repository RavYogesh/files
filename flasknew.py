Here's the Python code that achieves what you described:

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

# Function to find top 5 similar utterances and their respective intents from unique intents
def find_top_similar(keyword):
    keyword_vector = tfidf_vectorizer.transform([keyword])
    all_utterances = df['utterance'].tolist()
    all_intents = df['intent'].tolist()
    unique_intents = set(all_intents)
    top_utterances = []

    for intent in unique_intents:
        intent_indices = [i for i, x in enumerate(all_intents) if x == intent]
        intent_matrix = tfidf_matrix[intent_indices]
        cosine_similarities = cosine_similarity(keyword_vector, intent_matrix)
        top_indices = cosine_similarities.argsort()[0][-5:][::-1]
        top_utterances.extend([(all_utterances[intent_indices[idx]], intent) for idx in top_indices])

    return top_utterances

# Set up Flask server
app = Flask(__name__)

@app.route('/find_similar', methods=['POST'])
def find_similar_handler():
    request_data = request.json
    keyword = request_data.get('keyword', '')
    top_utterances = find_top_similar(keyword)
    response_data = {'top_utterances': top_utterances}
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
```

This code reads the Excel file, calculates the TF-IDF matrix, and then finds the top 5 similar utterances from unique intents that match the given keyword. The Flask server exposes an endpoint `/find_similar` that accepts POST requests with JSON data containing the keyword. It then returns a JSON response containing the top matching utterances and their respective intents.