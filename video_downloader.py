from yt_dlp import YoutubeDL

url = "https://www.youtube.com/watch?v=6llQB4p9NT4"

ydl_opts = {
    "format": "best",
}

with YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
