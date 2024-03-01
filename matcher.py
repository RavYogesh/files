import spacy
from spacy.matcher import Matcher

def extract_final_amount(doc):
  """
  Matches the pattern "final amount" followed by a number and returns the number.
  """
  matcher = Matcher(doc.vocab)
  pattern = [{"LOWER": "final"}, {"LOWER": "amount"}, {"NUM": True}]
  matcher.add("FINAL_AMOUNT", None, pattern)
  matches = matcher(doc)
  for match_id, start, end in matches:
    span = doc[start:end]
    return span.text.split()[-1]  # Extract the last element (number)

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(extract_final_amount, name="extract_final_amount")

# Example usage
doc = nlp("The final amount is $1,234.56.")
final_amount = doc._.extract_final_amount  # Access the custom attribute

print(final_amount)  # Output: $1,234.56
