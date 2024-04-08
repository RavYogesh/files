import requests

# Replace with your API endpoint URL
url = "https://api.example.com/endpoint"

# Create your data as a dictionary
data = {"key1": "value1", "key2": "value2"}

# Send the POST request with JSON data
response = requests.post(url, json=data)

# Check for successful response
if response.status_code == 200:
  # Process the JSON response
  print(response.json())
else:
  print(f"Error: API request failed with status code {response.status_code}")
