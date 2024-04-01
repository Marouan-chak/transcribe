import json
import shutil
import logging
import os
import subprocess
import sys
from datetime import datetime
from collections import deque
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from tafrigh import Config, TranscriptType, Writer, farrigh
from tafrigh.recognizers.wit_recognizer import WitRecognizer

app = Flask(__name__)
CORS(app)

def download_youtube_audio(youtube_url):
    # Generate a timestamp string for unique folder names
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # Update the output_path to use the timestamp in the folder name
    output_path = Path(__file__).parent / 'downloads' / timestamp / '%(id)s.%(ext)s'
    command = ['yt-dlp', '-x', '--audio-format', 'wav', '-o', str(output_path), youtube_url]
    subprocess.run(command, check=True)
    # Find the first .wav file in the newly created timestamp-named directory
    audio_file = next((Path(__file__).parent / 'downloads' / timestamp).glob('*.wav'), None)
    return audio_file, timestamp  # Return the timestamp for further use

def convert_video_to_audio(video_path, timestamp):
    # Use the same timestamp to store the converted audio
    audio_output_path = Path(__file__).parent / 'downloads' / timestamp / (video_path.stem + '.wav')
    command = ['ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', str(audio_output_path)]
    subprocess.run(command, check=True)
    print(f"Video converted to audio: {audio_output_path}")
    return audio_output_path

def transcribe_file(file_path, language_sign, api_key, timestamp):
    # Transcription should save files in the same timestamp-named directory
    output_dir = Path(__file__).parent / 'downloads' / timestamp
    config = Config(
        urls_or_paths=[str(file_path)],
        skip_if_output_exist=False,
        playlist_items="",
        verbose=False,
        model_name_or_path="",
        task="",
        language="",
        use_faster_whisper=False,
        beam_size=0,
        ct2_compute_type="",
        wit_client_access_tokens=[api_key], 
        max_cutting_duration=5,
        min_words_per_segment=1,
        save_files_before_compact=False,
        save_yt_dlp_responses=False,
        output_sample=0,
        output_formats=[TranscriptType.TXT, TranscriptType.SRT],
        output_dir=str(output_dir),
    )

    print(f"Transcribing file: {file_path}")
    progress = deque(farrigh(config), maxlen=0)
    print(f"Transcription completed. Check the output directory for the generated files.")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json()
    api_key = data.get('api_key')
    language_sign = data.get('language_sign')
    youtube_url = data.get('youtube_url')
    file_path = data.get('file_path')

    if not api_key:
        return jsonify({'error': 'API key is required'}), 401
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  # Prepare a timestamp for naming

    if youtube_url:
        audio_file, timestamp = download_youtube_audio(youtube_url)
        if audio_file:
            transcribe_file(audio_file, language_sign, api_key, timestamp)
        else:
            return jsonify({'error': 'Failed to download or process the YouTube video'}), 500
    elif file_path:
        file_path = Path(file_path)
        audio_file = None
        if file_path.suffix.lower() in ['.mp3', '.mp4', '.mkv', '.avi']:
            # Use the timestamp when converting to ensure unique folder creation
            audio_file = convert_video_to_audio(file_path, timestamp)
            transcribe_file(audio_file, language_sign, api_key, timestamp)
        if not audio_file:
            return jsonify({'error': 'Failed to process the local file'}), 500
    else:
        return jsonify({'error': 'No YouTube URL or file path provided'}), 400

    # Create a zip file of the timestamp-named folder
    downloads_path = Path(__file__).parent / 'downloads'
    zip_filename = f"{downloads_path}/transcription_results_{timestamp}.zip"
    zip_command = ['zip', '-r', zip_filename, downloads_path / timestamp]
    subprocess.run(zip_command, check=True)

    response = send_file(zip_filename, as_attachment=True)
    response.headers['Content-Disposition'] = 'attachment; filename=transcription_results.zip'

    # Delete the downloads folder after sending the zip file
    shutil.rmtree(downloads_path / timestamp)

    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
