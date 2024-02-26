import pandas as pd
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Read the text document into a DataFrame
df = pd.read_csv("your_text_file.csv")

# Define a function to process text and extract entities
def extract_entities(text):
    doc = nlp(text)
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    amounts = [ent.text for ent in doc.ents if ent.label_ == "MONEY"]
    return dates, amounts

# Create new columns for extracted dates and amounts
df["Dates"] = df["text"].apply(extract_entities)[0]
df["Amounts"] = df["text"].apply(extract_entities)[1]

# Drop the original text column if not needed
if not df["text"].isnull().any():
    df.drop("text", axis=1, inplace=True)

# Write the DataFrame to a CSV file
df.to_csv("extracted_data.csv", index=False)

print("Dates and amounts extracted and written to extracted_data.csv")
