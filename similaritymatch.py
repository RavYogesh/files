You can achieve this using Python's pandas library for reading Excel files and a similarity matching function like cosine similarity from the scikit-learn library. Here's a basic outline of how you can do it:

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read the Excel file
df = pd.read_excel('your_excel_file.xlsx')

# Define the keyword
keyword = "your_keyword"

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer and transform the data
tfidf_matrix = tfidf_vectorizer.fit_transform(df['utterance'])

# Calculate the cosine similarity between the keyword and all utterances
keyword_vector = tfidf_vectorizer.transform([keyword])
cosine_similarities = cosine_similarity(keyword_vector, tfidf_matrix)

# Get the indices of the top 5 similar utterances
top_indices = cosine_similarities.argsort()[0][-5:][::-1]

# Print the top 5 similar utterances
for idx in top_indices:
    print(df['utterance'][idx])
```

Make sure to replace `'your_excel_file.xlsx'` with the path to your Excel file, and adjust column names accordingly if needed. Additionally, you might need to install scikit-learn and pandas if you haven't already:

```
pip install scikit-learn pandas
```

This code will print the top 5 utterances from your Excel file that are most similar to the given keyword.