import os
import requests

from dotenv import load_dotenv


# Load .env
load_dotenv()


API_KEY = os.getenv("YOUTUBE_API_KEY")

BASE_URL = "https://www.googleapis.com/youtube/v3"


# Check API key
if not API_KEY:

    print("WARNING: YOUTUBE_API_KEY was not found in .env")

else:

    print("YouTube API key loaded successfully.")


def get_channel_by_id(channel_id):

    url = f"{BASE_URL}/channels"

    params = {
        "part": "snippet,statistics,contentDetails",
        "id": channel_id,
        "key": API_KEY
    }

    response = requests.get(
        url,
        params=params
    )

    if response.status_code != 200:

        raise Exception(
            response.json()
        )

    data = response.json()

    if not data.get("items"):

        return None

    return data["items"][0]


def get_channel_by_handle(handle):

    url = f"{BASE_URL}/channels"

    params = {
        "part": "snippet,statistics,contentDetails",
        "forHandle": handle,
        "key": API_KEY
    }

    response = requests.get(
        url,
        params=params
    )

    if response.status_code != 200:

        raise Exception(
            response.json()
        )

    data = response.json()

    if not data.get("items"):

        return None

    return data["items"][0]


def get_recent_videos(
    uploads_playlist_id,
    max_results=20
):

    url = f"{BASE_URL}/playlistItems"

    params = {
        "part": "snippet,contentDetails",
        "playlistId": uploads_playlist_id,
        "maxResults": max_results,
        "key": API_KEY
    }

    response = requests.get(
        url,
        params=params
    )

    if response.status_code != 200:

        raise Exception(
            response.json()
        )

    data = response.json()

    return data.get("items", [])


def get_video_statistics(video_ids):

    if not video_ids:

        return []


    url = f"{BASE_URL}/videos"

    params = {
        "part": "snippet,statistics",
        "id": ",".join(video_ids),
        "key": API_KEY
    }

    response = requests.get(
        url,
        params=params
    )

    if response.status_code != 200:

        raise Exception(
            response.json()
        )

    data = response.json()

    return data.get("items", [])
