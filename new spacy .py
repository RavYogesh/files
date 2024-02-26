import spacy
import csv

# Load the spaCy model and text document
nlp = spacy.load("en_core_web_sm")
with open("your_text_file.txt", "r") as f:
    text = f.read()

# Process the text and extract entities
doc = nlp(text)
dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
amounts = [ent.text for ent in doc.ents if ent.label_ == "MONEY"]

# Write extracted data to CSV file
with open("extracted_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Dates", "Amounts"])
    writer.writerows(zip(dates, amounts))

print("Dates and amounts extracted and written to extracted_data.csv")
