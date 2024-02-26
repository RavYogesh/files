import spacy
import csv

# Load English tokenizer, tagger, parser, and NER
nlp = spacy.load("en_core_web_sm")

# ocr = "my name is ravi. i transferred 100 dollars on 23/12/2012 to my friend"
# text2 = nlp(ocr)
# for word in text2.ents:
#     print(word.text,word.label_)


import pandas as pd

# Read the text file and split it into paragraphs
with open("ocr2.txt", "r") as file:
    text = file.read()

paragraphs = text.split("\n\n")  # Assuming paragraphs are separated by two newline characters

# Take the first 5 paragraphs
selected_paragraphs = paragraphs[:5]

# Create a Pandas Series from the selected paragraphs
series = pd.Series(selected_paragraphs)

# Print the Series
print(series)
docs = list(nlp.pipe(series))
from itertools import groupby
parsed_data = []
for doc in docs:
    doc_dict = {key: list(set(map(lambda x: str(x), g)))
                for key, g
                in groupby(sorted(doc.ents, key=lambda x: x.label_), lambda x: x.label_) }
    parsed_data.append(doc_dict)

df = pd.DataFrame(parsed_data)
df.to_csv('output4.csv')

