import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables (for your API key)
load_dotenv()
api_key = os.getenv("HACKRX_API_KEY") # Ensure this is the correct key name

# The URL to your deployed Django API
url = "http://127.0.0.1:8000/api/v1/hackrx/run"

# The platform's sample request body uses "documents" and "questions"
# We must replicate this structure exactly in our test.
# The 'documents' key now holds a URL string, and 'questions' holds a list of strings.
questions_to_ask = [
    'What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?',
    'What is the waiting period for pre-existing diseases (PED) to be covered?',
    'Does this policy cover maternity expenses, and what are the conditions?',
    'What is the waiting period for cataract surgery?',
    'Are the medical expenses for an organ donor covered under this policy?',
    'What is the No Claim Discount (NCD) offered in this policy?',
    'Is there a benefit for preventive health check-ups?',
    'How does the policy define a \'Hospital\'?',
    'What is the extent of coverage for AYUSH treatments?',
    'Are there any sub-limits on room rent and ICU charges for Plan A?'
]

# Construct the payload exactly as the platform sends it
payload = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": questions_to_ask
}

# The platform requires an 'Authorization' header with a Bearer token.
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}" # Your API key goes here
}

print("Sending test request...")
try:
    # Make the POST request with the correct URL, headers, and payload
    res = requests.post(url, headers=headers, json=payload, timeout=30)
    
    print(f"Status Code: {res.status_code}")
    
    # Try to print the JSON response, but handle cases where it's not JSON
    try:
        print("Response JSON:")
        print(json.dumps(res.json(), indent=2))
    except json.JSONDecodeError:
        print("Response Content (not JSON):")
        print(res.text)
    
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

