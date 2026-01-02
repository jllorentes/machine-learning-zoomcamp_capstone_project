import requests
import os
import sys

# --- CONFIGURACIÓN ---
URL = "http://127.0.0.1:9696/predict"

IMAGE_PATH = "./data/processed/test/damage/0042.JPEG" 

print(f"Starting test against: {URL}")

if not os.path.exists(IMAGE_PATH):
    print(f"I couldn't find {IMAGE_PATH}. Looking for an alternative image....")
    found = False
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                IMAGE_PATH = os.path.join(root, file)
                print(f"Alternative found: {IMAGE_PATH}")
                found = True
                break
        if found: break
    
    if not found:
        print("Error: I cannot find any .jpg images in the data folder./")
        sys.exit(1)

print(f"Sending image: {IMAGE_PATH}...")
try:
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": (os.path.basename(IMAGE_PATH), f, "image/jpeg")}
        response = requests.post(URL, files=files)
    
    print(f"📡 Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("\n SERVER RESPONSE:")
        print(response.json())
    else:
        print("\n SERVER ERROR:")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("\n Error: Unable to connect to the API.")
    print("Is the server running? (uv run uvicorn predict:app ...)")
except Exception as e:
    print(f"\n Unexpected error: {e}")