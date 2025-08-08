import requests

url = "http://localhost:8000/api/v1/hackrx/run"
payload = {
    "query": "A 50-year-old male is hospitalized for stroke and has an HDFC ERGO 'EASY HEALTH' policy."
}

res = requests.post(url, json=payload)
print(res.status_code)
print(res.json())
