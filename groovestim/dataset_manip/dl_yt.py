import yt_dlp
from yt_dlp.utils import download_range_func


def ytdl(url,timestamp,duration,outpath='./test_dl_yt_axel'):
    minutes, seconds = timestamp.strip().split(':')

    # Convert minutes and seconds to integers
    minutes = int(minutes)
    seconds = int(seconds)
    
    # Convert minutes to seconds and add to the total seconds
    start = minutes * 60 + seconds
    
    stop = start + duration

    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'paths' : {'home':outpath},
        'download_ranges' : download_range_func(None,[(start,stop)]),
        'force_keyframes_at_cuts': True, # for yt links,
        # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
        'postprocessors': [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download([url])

url = 'https://www.youtube.com/watch?v=GZlVT8gPGEs'

timestamp = '  0:10 '
duration = 5

ytdl(url,timestamp='0:20',duration=10,outpath='./yt')