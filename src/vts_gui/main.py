# Veil's Transcription/Translation Service
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import queue
import os
import sys
import traceback
import time
import torch

# Import pipelines
from .pipeline.transcription_pipeline import TranscriptionPipeline
from .pipeline.translation_pipeline import TranslationPipeline
from .pipeline.postprocessing_pipeline import PostProcessingPipeline
from .pipeline.utils import (
    load_model_info,
    load_audio_duration,
    format_timestamp,
    format_duration,
    language_code_to_name,
    language_name_to_code,
    MODELS_WITH_EN,
    LANGUAGES
)

import importlib.resources as pkg_resources
import vts_gui

task_progress_messages = {
    'transcribe': 'Transcribing',
    'translate': 'Translating'
}

task_completed_messages = {
    'transcribe': 'Transcription completed.',
    'translate': 'Translation completed.'
}

class WhisperApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VTS GUI v1.2.2")
        self.geometry("600x700")

        with pkg_resources.path(vts_gui, "favicon.ico") as icon_path:
            self.iconbitmap(icon_path)

        self.queue = queue.Queue()
        self.models = ["tiny", "base", "small", "medium", "large", "turbo"]
        self.models_requirements = {
            "tiny": {"ram": 1, "vram": 1},
            "base": {"ram": 1, "vram": 1},
            "small": {"ram": 2, "vram": 2},
            "medium": {"ram": 5, "vram": 5},
            "large": {"ram": 10, "vram": 10},
            "turbo": {"ram": 6, "vram": 6},
        }

        self.create_widgets()
        self.check_queue()

    def create_widgets(self):
        # File Selection
        self.file_label = tk.Label(self, text="Source Media:")
        self.file_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)

        self.file_entry = tk.Entry(self, width=50)
        self.file_entry.grid(row=0, column=1, padx=10, pady=10)

        self.browse_button = tk.Button(self, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=10, pady=10)

        # Model Selection
        self.model_label = tk.Label(self, text="Model:")
        self.model_label.grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)

        self.model_var = tk.StringVar(value=self.models[0])
        self.model_menu = tk.OptionMenu(self, self.model_var, *self.models, command=self.update_controls)
        self.model_menu.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)

        self.model_info_label = tk.Label(self, text="")
        self.model_info_label.grid(row=1, column=2, padx=10, pady=10, sticky=tk.W)

        # English Only checkbox
        self.english_only_var = tk.BooleanVar()
        self.english_only_check = tk.Checkbutton(
            self, text="English Only", variable=self.english_only_var, command=self.update_controls
        )
        self.english_only_check.grid(row=2, column=1, padx=10, pady=10, sticky=tk.W)

        # Language selection
        self.lang_label = tk.Label(self, text="Language:")
        self.lang_label.grid(row=3, column=0, sticky=tk.W, padx=10, pady=10)

        self.lang_var = tk.StringVar(value="Auto Detect")
        self.lang_menu = tk.OptionMenu(self, self.lang_var, *LANGUAGES)
        self.lang_menu.grid(row=3, column=1, padx=10, pady=10, sticky=tk.W)

        # Task selection
        self.task_label = tk.Label(self, text="Task:")
        self.task_label.grid(row=4, column=0, sticky=tk.W, padx=10, pady=10)

        self.task_var = tk.StringVar(value="Transcribe")
        self.task_var.trace('w', self.update_controls)
        self.task_menu = tk.OptionMenu(self, self.task_var, "Transcribe", "Translate")
        self.task_menu.grid(row=4, column=1, padx=10, pady=10, sticky=tk.W)

        # Precision selection
        self.precision_label = tk.Label(self, text="Precision:")
        self.precision_label.grid(row=5, column=0, sticky=tk.W, padx=10, pady=10)

        self.precision_var = tk.StringVar(value="FP16")
        self.precision_menu = tk.OptionMenu(self, self.precision_var, "FP16", "FP32")
        self.precision_menu.grid(row=5, column=1, padx=10, pady=10, sticky=tk.W)

        # Device selection
        self.device_label = tk.Label(self, text="Device:")
        self.device_label.grid(row=6, column=0, sticky=tk.W, padx=10, pady=10)

        self.available_devices = []
        if torch.cuda.is_available():
            self.available_devices.extend([f"GPU {i}: {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())])
            self.available_devices.append("CPU")
            default_device = self.available_devices[0]
        else:
            self.available_devices.append("CPU")
            default_device = "CPU"

        self.device_var = tk.StringVar(value=default_device)
        self.device_menu = tk.OptionMenu(self, self.device_var, *self.available_devices, command=self.update_controls)
        self.device_menu.grid(row=6, column=1, padx=10, pady=10, sticky=tk.W)

        # Start and Save Buttons
        self.start_button = tk.Button(self, text="Start", command=self.start_processing)
        self.start_button.grid(row=7, column=1, padx=10, pady=10)

        self.save_button = tk.Button(self, text="Save Subtitles", command=self.save_output)
        self.save_button.grid(row=7, column=2, padx=10, pady=10)

        # Output Text
        self.output_text = tk.Text(self, wrap=tk.WORD)
        self.output_text.grid(row=8, column=0, columnspan=3, padx=10, pady=10)

        self.grid_rowconfigure(8, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.update_controls()

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio/Video Files", "*.mp3 *.wav *.flac *.m4a *.mp4 *.mkv *.avi *.mov *.webm *.aac *.ogg *.wma")]
        )
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def update_controls(self, *args):
        task = self.task_var.get().lower()
        selected_device = self.device_var.get()
        model_name = self.model_var.get()

        # Update model info display
        model_info = load_model_info(model_name)
        if model_info:
            self.model_info_label.config(text=f"RAM: {model_info['ram']}, Speed: {model_info['speed']}")

        # Enable/Disable English only checkbox
        if task == "translate":
            self.english_only_check.config(state=tk.DISABLED)
            self.english_only_var.set(False)
        elif model_name in MODELS_WITH_EN:
            self.english_only_check.config(state=tk.NORMAL)
        else:
            self.english_only_check.config(state=tk.DISABLED)
            self.english_only_var.set(False)

        # Update language menu
        menu = self.lang_menu["menu"]
        menu.delete(0, "end")
        if task == "translate":
            # For translation, do not allow 'Auto Detect'
            for lang in LANGUAGES:
                if lang != "Auto Detect":
                    menu.add_command(label=lang, command=tk._setit(self.lang_var, lang))
            if self.lang_var.get() == "Auto Detect":
                # Set a default language if needed
                self.lang_var.set("English")  # or any common language
        else:
            for lang in LANGUAGES:
                menu.add_command(label=lang, command=tk._setit(self.lang_var, lang))
            if self.english_only_var.get():
                self.lang_menu.config(state=tk.DISABLED)
            else:
                self.lang_menu.config(state=tk.NORMAL)

        # Update Precision based on device
        if selected_device == "CPU":
            # Only FP32 on CPU
            self.precision_var.set("FP32")
            self.precision_menu["menu"].delete(0, "end")
            self.precision_menu["menu"].add_command(label="FP32", command=tk._setit(self.precision_var, "FP32"))
            self.precision_menu["menu"].add_command(label="FP16 (Not available on CPU)", state="disabled")
        else:
            # FP16 and FP32 available on GPU
            self.precision_menu["menu"].delete(0, "end")
            self.precision_menu["menu"].add_command(label="FP16", command=tk._setit(self.precision_var, "FP16"))
            self.precision_menu["menu"].add_command(label="FP32", command=tk._setit(self.precision_var, "FP32"))

    def start_processing(self):
        audio_path = self.file_entry.get()
        if not audio_path or not os.path.isfile(audio_path):
            messagebox.showerror("Error", "Please select a valid audio/video file.")
            return

        task = self.task_var.get().lower()
        model_name = self.model_var.get()
        language = self.lang_var.get()
        english_only = self.english_only_var.get()
        precision = self.precision_var.get()
        selected_device = self.device_var.get()
        
        # Adjust model name if English Only is selected and model supports it
        if english_only and model_name in MODELS_WITH_EN:
            model_name += ".en"
        
        fp16 = (precision == "FP16")

        if task == "translate" and language == "Auto Detect":
            messagebox.showerror("Error", "Please select a language for translation.")
            return

        self.start_button.config(state=tk.DISABLED)
        self.output_text.delete(1.0, tk.END)
        self.queue.put("Processing...\n")

        threading.Thread(target=self.run_processing, args=(audio_path, model_name, language, task, fp16, selected_device)).start()

    def run_processing(self, audio_path, model_name, language, task, fp16, selected_device):
        try:
            start_time = time.time()
            # Determine device
            if selected_device == "CPU":
                device = "cpu"
                self.queue.put("Using CPU.\n")
            else:
                try:
                    device_index = int(selected_device.split()[1][:-1])
                    device = f"cuda:{device_index}"
                    gpu_name = torch.cuda.get_device_name(device_index)
                    self.queue.put(f"Using GPU {device_index}: {gpu_name}\n")
                except:
                    device = "cpu"
                    self.queue.put("Invalid GPU selection. Defaulting to CPU.\n")

            language_code = language_name_to_code.get(language, None)
            
            # Create and run the appropriate pipeline
            if task == "transcribe":
                pipeline = TranscriptionPipeline(model_name, device, fp16, language_code=language_code)
            else:
                # For translation, language_code should not be None. If it is, you handle it.
                pipeline = TranslationPipeline(model_name, device, fp16, language_code=language_code)

            self.queue.put(f"{task_progress_messages.get(task, 'Processing')}...\n")

            segments, detected_language = pipeline.run(audio_path)
            
            end_time = time.time()
            transcription_time = end_time - start_time
            audio_duration = load_audio_duration(audio_path)

            if language_code is None:
                # Language was auto-detected in transcription mode
                detected_language_name = language_code_to_name.get(detected_language, detected_language).title()
                language_message = f"Detected language: {detected_language_name}"
            else:
                language_message = f"Language: {language}"

            self.segments = segments

            self.queue.put("\n")
            self.queue.put(f"{language_message}\n\n")
            completion_message = task_completed_messages.get(task, 'Processing completed.')
            self.queue.put(f"{completion_message}\n")
            self.queue.put("You can now save the subtitles.\n\n")

            # Efficiency report
            if audio_duration and transcription_time:
                processing_time_str = format_duration(transcription_time)
                audio_duration_str = format_duration(audio_duration)
                speed = round(audio_duration / transcription_time, 2)
                speed_str = f"Speed: ~{speed}x"

                # Compute avg_logprob if available
                avg_logprob = None
                try:
                    avg_logprob = sum(s.get('avg_logprob', 0) for s in segments if 'avg_logprob' in s) / len(segments)
                except:
                    pass

                # Confidence estimate
                if avg_logprob is not None:
                    if avg_logprob >= -0.2:
                        confidence = "Perfection"
                    elif avg_logprob >= -0.5:
                        confidence = "Very High"
                    elif avg_logprob >= -0.8:
                        confidence = "High"
                    elif avg_logprob >= -1.0:
                        confidence = "Moderate"
                    else:
                        confidence = "Low"
                else:
                    confidence = "N/A"

                total_words = len(' '.join([s['text'] for s in segments]).split())
                total_segments = len(segments)

                # report back our processing details to be logged
                efficiency_report = (
                    f"Processing time: {processing_time_str}\n"
                    f"Audio duration: {audio_duration_str}\n"
                    f"{speed_str}\n"
                    f"Confidence: {confidence}\n"
                    f"Total segments: {total_segments}\n"
                    f"Total words: {total_words}"
                )

                self.queue.put("Efficiency Report:\n")
                self.queue.put(efficiency_report + "\n")

        except Exception as e:
            tb = traceback.format_exc()
            self.queue.put(f"Error: {e}\n{tb}")
            self.segments = None
        finally:
            self.queue.put(('ENABLE_START_BUTTON', None))

    def save_output(self):
        if not hasattr(self, 'segments') or not self.segments:
            messagebox.showwarning("Warning", "No subtitles available to save.")
            return

        media_filename = os.path.splitext(os.path.basename(self.file_entry.get()))[0]
        lang_selected = self.lang_var.get()
        language_code = language_name_to_code.get(lang_selected, "auto")

        task = self.task_var.get().lower()
        if task == "transcribe":
            default_srt_name = f"{media_filename}.{language_code}.srt"
        else:
            default_srt_name = f"{media_filename}.en.srt"

        directory = filedialog.askdirectory()
        if directory:
            file_path = os.path.join(directory, default_srt_name)
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    for i, segment in enumerate(self.segments, start=1):
                        start = format_timestamp(segment['start'])
                        end = format_timestamp(segment['end'])
                        f.write(f"{i}\n{start} --> {end}\n{segment['text'].strip()}\n\n")
                    self.add_credits(f, i+1, self.segments[-1]['end'], task)
                messagebox.showinfo("Success", "Subtitles saved")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save subtitles: {e}")

    def add_credits(self, file_handle, index, start_time, task):
        end_time = start_time + 5
        start = format_timestamp(start_time + 1)
        end = format_timestamp(end_time)
        if task == "transcribe":
            credit_text = "Transcribed by VTS Service | Contact: veilbreaker@voidmaster.xyz"
        else:
            credit_text = "Translated by VTS Service | Contact: veilbreaker@voidmaster.xyz"
        file_handle.write(f"{index}\n{start} --> {end}\n{credit_text}\n\n")

    def check_queue(self):
        try:
            while True:
                message = self.queue.get_nowait()
                if isinstance(message, tuple):
                    command = message
                    if command == 'ENABLE_START_BUTTON':
                        self.start_button.config(state=tk.NORMAL)
                else:
                    self.output_text.insert(tk.END, message)
        except queue.Empty:
            pass
        self.after(100, self.check_queue)

def main():
    # Check command line arguments
    if "--version" in sys.argv:
        # Print version and exit
        from . import __version__
        print(__version__)
        return

    # Otherwise, launch the application
    app = WhisperApp()
    app.mainloop()

if __name__ == "__main__":
    main()
