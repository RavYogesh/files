import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

def generate():
  vertexai.init(project="solar-theory-417319", location="us-central1")
  model = GenerativeModel("gemini-1.0-pro-vision-001")
  responses = model.generate_content(
      [text1],
      generation_config=generation_config,
      safety_settings=safety_settings,
      stream=True,
  )

  for response in responses:
    print(response.text, end="")

text1 = """Can you generate 20 test utterances to test regex for an entity with following requirement: \"it has to be min. 6 characters long with minimum 2 numbers and 1 special character\" . Please give only valid and meaningful results clearly separated between positive and negative scenarios. Please exhaustively cover all the corner cases possible for the negative test cases."""

generation_config = {
    "max_output_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.4,
    "top_k": 32,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

generate()
