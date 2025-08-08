import requests

# url = "https://bajajapi3.onrender.com/api/v1/hackrx/run"
url = "http://localhost:8000/api/v1/hackrx/run"
payload = {
    "query": "A 50-year-old male is hospitalized for stroke and has an HDFC ERGO 'EASY HEALTH' policy."
}
print("now")
res = requests.post(url, json=payload)
print(res.status_code)
print(res.json())
