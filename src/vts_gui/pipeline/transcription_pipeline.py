# pipeline/transcription_pipeline.py
import whisper

class TranscriptionPipeline:
    def __init__(self, model_name, device, fp16, language_code=None):
        self.model_name = model_name
        self.device = device
        self.fp16 = fp16
        self.language_code = language_code
        self.model = whisper.load_model(model_name, device=device)
    
    def run(self, audio_path):
        # Run Whisper with task='transcribe'
        result = self.model.transcribe(
            audio_path, 
            language=self.language_code, 
            task="transcribe", 
            fp16=self.fp16
        )
        segments = result["segments"]
        detected_language = result["language"]
        return segments, detected_language
