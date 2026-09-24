from datetime import datetime, timezone
import re
import os

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from youtube_api import (
    get_channel_by_id,
    get_channel_by_handle,
    get_recent_videos,
    get_video_statistics,
)
from detector import analyze_channel

load_dotenv()

app = Flask(__name__)


def extract_channel_info(url):
    url = url.strip()

    # UC... direct channel ID
    if re.match(r'^UC[\w-]{22}$', url):
        return {"type": "id", "value": url}

    # /channel/UC...
    m = re.search(r'youtube\.com/channel/(UC[\w-]{22})', url)
    if m:
        return {"type": "id", "value": m.group(1)}

    # /@handle or @handle
    m = re.search(r'(?:youtube\.com/)?@([\w.-]+)', url)
    if m:
        return {"type": "handle", "value": m.group(1)}

    # /c/ or /user/
    m = re.search(r'youtube\.com/(?:c|user)/([\w.-]+)', url)
    if m:
        return {"type": "handle", "value": m.group(1)}

    # bare @
    if url.startswith('@'):
        return {"type": "handle", "value": url[1:]}

    return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received."}), 400

        channel_url = data.get("url", "").strip()
        if not channel_url:
            return jsonify({"error": "Please enter a YouTube channel URL."}), 400

        info = extract_channel_info(channel_url)
        if not info:
            return jsonify({
                "error": (
                    "Invalid YouTube URL. Use formats like: "
                    "youtube.com/@channel  or  youtube.com/channel/UC..."
                )
            }), 400

        # Fetch channel
        if info["type"] == "id":
            channel = get_channel_by_id(info["value"])
        else:
            channel = get_channel_by_handle(info["value"])

        if not channel:
            return jsonify({
                "error": "Channel not found. Please check the URL."
            }), 404

        # Fetch recent videos
        uploads_playlist = (
            channel
            .get("contentDetails", {})
            .get("relatedPlaylists", {})
            .get("uploads")
        )
        videos = []
        if uploads_playlist:
            playlist_items = get_recent_videos(uploads_playlist, 20)
            video_ids = [
                item.get("contentDetails", {}).get("videoId")
                for item in playlist_items
                if item.get("contentDetails", {}).get("videoId")
            ]
            if video_ids:
                videos = get_video_statistics(video_ids)

        # Run Isolation Forest ML detector
        analysis = analyze_channel(channel, videos)

        # Build response
        snippet    = channel.get("snippet", {})
        statistics = channel.get("statistics", {})
        thumbnails = snippet.get("thumbnails", {})
        thumb_url  = (
            thumbnails.get("medium", {}).get("url")
            or thumbnails.get("default", {}).get("url", "")
        )

        result = {
            "channel": {
                "id":          channel.get("id", ""),
                "name":        snippet.get("title", "Unknown Channel"),
                "description": snippet.get("description", "")[:200],
                "thumbnail":   thumb_url,
                "country":     snippet.get("country", ""),
                "handle":      snippet.get("customUrl", ""),
                "subscribers": statistics.get("subscriberCount", "0"),
                "views":       statistics.get("viewCount", "0"),
                "videos":      statistics.get("videoCount", "0"),
            },
            "analysis": analysis,
        }

        return jsonify(result)

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": "Something went wrong: " + str(e)}), 500


if __name__ == "__main__":
    print("\n========================================")
    print("   YT BotScope — Isolation Forest ML")
    print("========================================")
    print("\nOpen: http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
