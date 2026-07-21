"""
Debug script to fetch exact 500 error detail from running uvicorn server.
"""
import urllib.request
import urllib.error

url = "http://127.0.0.1:8000/api/v1/recommendations/personalized?history=B0B4HJNPV4,B0B1YZX72F,B08L879JSN,B0B9959XF3,B0BC9BW512,B0B2RBP83P,B09PTT8DZF,B08LW31NQ6,B09P22HXH6,B0B1YY6JJL,B0B997FBZT,B00EDJJ7FS,B08BJN4MP3,B09XXZXQC1,B096MSW6CT&k=8"

try:
    print(f"Fetching {url} ...")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        print(f"STATUS: {resp.status}")
        print("BODY:", resp.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print(f"HTTP ERROR: {e.code}")
    err_body = e.read().decode('utf-8')
    print("ERROR BODY:", err_body)
except Exception as e:
    print(f"OTHER ERROR: {e}")
