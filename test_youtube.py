import os
import requests
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("YOUTUBE_API_KEY")

print("API key loaded:", bool(key))

if not key:
    print("❌ API key is not loaded from .env")
    exit()

url = "https://www.googleapis.com/youtube/v3/channels"

params = {
    "part": "snippet,statistics",
    "id": "UC_x5XG1OV2P6uZZ5FSM9Ttw",
    "key": key
}

response = requests.get(url, params=params)

print("Status:", response.status_code)

if response.status_code == 200:
    print("✅ YouTube API is working!")
    print(response.json()["items"][0]["snippet"]["title"])
else:
    print("❌ YouTube API error:")
    print(response.text)