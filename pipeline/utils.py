# pipeline/utils.py
import datetime
import subprocess

# Language code maps from Whisper (already defined in original code)
import whisper

language_code_to_name = whisper.tokenizer.LANGUAGES
language_name_to_code = {v.title(): k for k, v in language_code_to_name.items()}
LANGUAGES = sorted(language_name_to_code.keys())
LANGUAGES.insert(0, 'Auto Detect')
MODELS_WITH_EN = ["tiny", "base", "small", "medium"]

MODEL_INFO = {
    "tiny": {"ram": "~1 GB", "speed": "~10x"},
    "base": {"ram": "~1 GB", "speed": "~7x"},
    "small": {"ram": "~2 GB", "speed": "~4x"},
    "medium": {"ram": "~5 GB", "speed": "~2x"},
    "large": {"ram": "~10 GB", "speed": "1x"},
    "turbo": {"ram": "~6 GB", "speed": "~8x"},
}

def load_model_info(model_name):
    return MODEL_INFO.get(model_name, None)

def load_audio_duration(audio_path):
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        audio_path
    ]
    try:
        ffprobe_result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration = float(ffprobe_result.stdout.strip())
        return duration
    except:
        return None

def format_timestamp(seconds):
    delta = datetime.timedelta(seconds=seconds)
    timestamp = (datetime.datetime(1,1,1) + delta).strftime("%H:%M:%S,%f")[:-3]
    return timestamp

def format_duration(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"
