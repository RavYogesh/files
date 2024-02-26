import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Load your text document
with open("your_text_file.txt", "r") as f:
    text = f.read()

# Process the text
doc = nlp(text)

# Extract dates and amounts
dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
amounts = [ent.text for ent in doc.ents if ent.label_ == "MONEY"]

# Print the extracted information
print("Dates:", dates)
print("Amounts:", amounts)
