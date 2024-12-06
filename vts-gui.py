# Veil's Transcription/Translation Service v1.1
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import whisper
import torch
import os
import psutil
import datetime
import time
import subprocess
import traceback
import queue

# Get the list of languages from whisper.tokenizer
language_code_to_name = whisper.tokenizer.LANGUAGES
language_name_to_code = {v.title(): k for k, v in language_code_to_name.items()}
languages = sorted(language_name_to_code.keys())
languages.insert(0, 'Auto Detect')  # Include 'Auto Detect' for transcription

# Dictionaries to specify progress and completion messages
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
        self.title("VTS GUI v1.1")
        self.geometry("600x700")
        
        # Set the window icon
        icon_path = "favicon.ico"
        if os.path.exists(icon_path):
            self.iconbitmap(icon_path)
        
        # Prepare languages without 'Auto Detect' for translation
        self.languages_no_auto = [lang for lang in languages if lang != "Auto Detect"]
        
        # Initialize a queue for thread-safe GUI updates
        self.queue = queue.Queue()
        self.create_widgets()
        self.check_queue()  # Start checking the queue

    def create_widgets(self):
        # Source file selection
        self.file_label = tk.Label(self, text="Source Media:")
        self.file_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)

        self.file_entry = tk.Entry(self, width=50)
        self.file_entry.grid(row=0, column=1, padx=10, pady=10)

        self.browse_button = tk.Button(self, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=10, pady=10)

        # Model selection
        self.model_label = tk.Label(self, text="Model:")
        self.model_label.grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)

        self.models_with_en = ["tiny", "base", "small", "medium"]
        models = ["tiny", "base", "small", "medium", "large", "turbo"]
        self.model_info = {
            "tiny": {"ram": "~1 GB", "speed": "~10x"},
            "base": {"ram": "~1 GB", "speed": "~7x"},
            "small": {"ram": "~2 GB", "speed": "~4x"},
            "medium": {"ram": "~6 GB", "speed": "~2x"},
            "large": {"ram": "~20 GB", "speed": "1x"},
            "turbo": {"ram": "~6 GB", "speed": "~8x"},
        }
        self.model_requirements = {
            "tiny": {"ram": 1, "vram": 1},
            "base": {"ram": 2, "vram": 2},
            "small": {"ram": 3, "vram": 3},
            "medium": {"ram": 6, "vram": 5},
            "large": {"ram": 20, "vram": 10},
            "turbo": {"ram": 6, "vram": 5},
        }
        self.model_var = tk.StringVar()
        self.model_menu = tk.OptionMenu(self, self.model_var, *models, command=self.update_controls)
        self.model_menu.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)

        # Model info label
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
        self.lang_menu = tk.OptionMenu(self, self.lang_var, *languages)
        self.lang_menu.grid(row=3, column=1, padx=10, pady=10, sticky=tk.W)

        # Task selection
        self.task_label = tk.Label(self, text="Task:")
        self.task_label.grid(row=4, column=0, sticky=tk.W, padx=10, pady=10)

        self.task_var = tk.StringVar(value="Transcribe")
        self.task_var.trace('w', self.update_controls)  # Update controls when task changes
        self.task_menu = tk.OptionMenu(self, self.task_var, "Transcribe", "Translate")
        self.task_menu.grid(row=4, column=1, padx=10, pady=10, sticky=tk.W)

        # Precision selection (FP16 or FP32)
        self.precision_label = tk.Label(self, text="Precision:")
        self.precision_label.grid(row=5, column=0, sticky=tk.W, padx=10, pady=10)

        self.precision_var = tk.StringVar(value="FP16")  # Default to FP16
        self.precision_menu = tk.OptionMenu(self, self.precision_var, "FP16", "FP32")
        self.precision_menu.grid(row=5, column=1, padx=10, pady=10, sticky=tk.W)

        # Device selection
        self.device_label = tk.Label(self, text="Device:")
        self.device_label.grid(row=6, column=0, sticky=tk.W, padx=10, pady=10)

        # Get the list of available devices
        self.available_devices = []
        if torch.cuda.is_available():
            self.available_devices.extend([f"GPU {i}: {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())])
            self.available_devices.append("CPU")  # Allow CPU as an option even if GPU is available
            default_device = self.available_devices[0]
        else:
            self.available_devices.append("CPU")
            default_device = "CPU"

        self.device_var = tk.StringVar(value=default_device)
        self.device_menu = tk.OptionMenu(self, self.device_var, *self.available_devices, command=self.update_controls)
        self.device_menu.grid(row=6, column=1, padx=10, pady=10, sticky=tk.W)

        # Start button
        self.start_button = tk.Button(self, text="Start", command=self.start_processing)
        self.start_button.grid(row=7, column=1, padx=10, pady=10)

        # Save button
        self.save_button = tk.Button(self, text="Save Subtitles", command=self.save_output)
        self.save_button.grid(row=7, column=2, padx=10, pady=10)

        # Output text box
        self.output_text = tk.Text(self, wrap=tk.WORD)
        self.output_text.grid(row=8, column=0, columnspan=3, padx=10, pady=10)

        # Configure grid weights
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
        task = self.task_var.get()
        selected_device = self.device_var.get()

        # Enable the model menu for transcription
        self.model_menu.config(state=tk.NORMAL)
        # Check available RAM and VRAM
        available_ram = psutil.virtual_memory().available / (1024 ** 3)  # In GB
        if selected_device == "CPU":
            available_vram = 0
        else:
            try:
                device_index = int(selected_device.split()[1][:-1])
                available_vram = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)  # In GB
            except (IndexError, ValueError):
                available_vram = 0
        # Disable models if insufficient RAM/VRAM
        models = ["tiny", "base", "small", "medium", "large", "turbo"]
        self.model_menu['menu'].delete(0, 'end')
        available_models = []
        for model in models:
            reqs = self.model_requirements[model]
            if selected_device == "CPU":
                # Exclude 'large' model on CPU due to high RAM requirements
                if model == 'large':
                    self.model_menu['menu'].add_command(
                        label=f"{model} (Not available on CPU)",
                        state="disabled"
                    )
                elif available_ram >= reqs['ram']:
                    self.model_menu['menu'].add_command(
                        label=model,
                        command=tk._setit(self.model_var, model, self.update_controls)
                    )
                    available_models.append(model)
                else:
                    self.model_menu['menu'].add_command(
                        label=f"{model} (Insufficient RAM)",
                        state="disabled"
                    )
            else:
                # Compare required VRAM
                if available_vram >= reqs['vram'] or task == "translate":
                    self.model_menu['menu'].add_command(
                        label=model,
                        command=tk._setit(self.model_var, model, self.update_controls)
                    )
                    available_models.append(model)
                else:
                    self.model_menu['menu'].add_command(
                        label=f"{model} (Insufficient VRAM)",
                        state="disabled"
                    )
        # Update model info label for the currently selected model
        if self.model_var.get() not in available_models:
            # Select the largest available model
            if available_models:
                self.model_var.set(available_models[-1])  # Choose the largest model available
            else:
                self.model_var.set("")  # No models available
        model_name = self.model_var.get()
        if model_name:
            ram = self.model_info[model_name]["ram"]
            speed = self.model_info[model_name]["speed"]
            self.model_info_label.config(text=f"RAM: {ram}, Speed: {speed}")
        else:
            self.model_info_label.config(text="No models available")

        # Update English Only checkbox
        if task.lower() == "translate":
            # Disable English Only checkbox
            self.english_only_check.config(state=tk.DISABLED)
            self.english_only_var.set(False)
        elif self.model_var.get() in self.models_with_en:
            # Enable English Only checkbox
            self.english_only_check.config(state=tk.NORMAL)
        else:
            # Disable English Only checkbox
            self.english_only_check.config(state=tk.DISABLED)
            self.english_only_var.set(False)

        # Update Language menu
        self.lang_menu['menu'].delete(0, 'end')
        if task.lower() == "translate":
            # For translation, do not allow 'Auto Detect'
            for lang in self.languages_no_auto:
                self.lang_menu['menu'].add_command(label=lang, command=tk._setit(self.lang_var, lang))
            # Ensure language menu is enabled
            self.lang_menu.config(state=tk.NORMAL)
            # If current language is 'Auto Detect', set to first language
            if self.lang_var.get() == "Auto Detect":
                self.lang_var.set(self.languages_no_auto[0] if self.languages_no_auto else "")
        else:
            # For transcription
            for lang in languages:
                self.lang_menu['menu'].add_command(label=lang, command=tk._setit(self.lang_var, lang))
            # Update Language menu based on 'English Only' checkbox
            if self.english_only_var.get():
                self.lang_menu.config(state=tk.DISABLED)
            else:
                self.lang_menu.config(state=tk.NORMAL)

        # Update Precision selection based on device selection
        if selected_device == "CPU":
            # Set precision to FP32 and disable FP16 option
            if self.precision_var.get() == "FP16":
                self.precision_var.set("FP32")
            self.precision_menu['menu'].delete(0, 'end')
            self.precision_menu['menu'].add_command(label="FP32", command=tk._setit(self.precision_var, "FP32"))
            # Disable FP16 option
            self.precision_menu['menu'].add_command(label="FP16 (Not available on CPU)", state="disabled")
            self.precision_menu.config(state=tk.NORMAL)
        else:
            # Enable both FP16 and FP32 options
            self.precision_menu['menu'].delete(0, 'end')
            self.precision_menu['menu'].add_command(label="FP16", command=tk._setit(self.precision_var, "FP16"))
            self.precision_menu['menu'].add_command(label="FP32", command=tk._setit(self.precision_var, "FP32"))
            self.precision_menu.config(state=tk.NORMAL)
            # Keep user's precision selection or default to FP16
            if self.precision_var.get() not in ["FP16", "FP32"]:
                self.precision_var.set("FP16")

    def start_processing(self):
        audio_path = self.file_entry.get()
        if not audio_path or not os.path.isfile(audio_path):
            messagebox.showerror("Error", "Please select a valid audio/video file.")
            return
        model_name = self.model_var.get()
        language = self.lang_var.get()
        task = self.task_var.get().lower()
        english_only = self.english_only_var.get()
        precision = self.precision_var.get()
        selected_device = self.device_var.get()

        # Adjust model name if English Only is selected
        if english_only and model_name in self.models_with_en:
            model_name += ".en"

        # Determine fp16 setting
        fp16 = precision == "FP16"

        if task == "translate" and language == "Auto Detect":
            messagebox.showerror("Error", "Please select a language for translation.")
            return

        # Disable the start button to prevent multiple clicks
        self.start_button.config(state=tk.DISABLED)
        self.output_text.delete(1.0, tk.END)
        self.queue.put("Processing...\n")

        # Run the process in a separate thread
        threading.Thread(
            target=self.run_processing, args=(audio_path, model_name, language, task, fp16, selected_device)
        ).start()

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
                except (IndexError, ValueError):
                    device = "cpu"
                    self.queue.put("Invalid GPU selection. Defaulting to CPU.\n")

            # Load model
            self.queue.put(f"Loading model '{model_name}' on {device}...\n")
            model = whisper.load_model(model_name, device=device)

            # Handle language option
            if task == "translate":
                # For translation, ensure language is specified
                language_code = language_name_to_code.get(language)
                if not language_code:
                    self.queue.put("Error: Invalid language selected for translation.\n")
                    self.queue.put(('ENABLE_START_BUTTON', None))
                    return
            else:
                # For transcription
                language_code = language_name_to_code.get(language) if language != "Auto Detect" else None

            # Transcribe or Translate the audio file
            progress_message = task_progress_messages.get(task, 'Processing')
            self.queue.put(f"{progress_message}...\n")

            result = model.transcribe(
                audio_path, language=language_code, task=task, fp16=fp16
            )

            self.segments = result["segments"]  # Save segments for SRT output
            detected_language = result["language"]
            detected_language_name = language_code_to_name.get(
                detected_language, detected_language
            ).title()

            end_time = time.time()
            self.transcription_time = end_time - start_time

            # Get duration of the audio file using ffprobe
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
                self.audio_duration = duration
            except Exception as e:
                self.queue.put(f"Could not determine audio duration: {e}\n")
                self.audio_duration = None

            # Compute average log probability if available
            try:
                avg_logprob = sum(s['avg_logprob'] for s in self.segments) / len(self.segments)
                self.avg_logprob = avg_logprob
            except KeyError:
                self.avg_logprob = None

            # Determine if the language was specified or auto-detected
            if language_code is None:
                language_message = f"Detected language: {detected_language_name}"
            else:
                language_message = f"Language: {detected_language_name}"

            # Update the output text box
            self.queue.put("\n")  # New line
            self.queue.put(f"{language_message}\n\n")
            completion_message = task_completed_messages.get(task, 'Processing completed.')
            self.queue.put(f"{completion_message}\n")
            self.queue.put("You can now save the subtitles.\n\n")

            # Generate efficiency report
            if hasattr(self, 'transcription_time') and self.audio_duration:
                processing_time = self.transcription_time
                audio_duration = self.audio_duration
                # Convert times to H:M:S format
                processing_time_str = self.format_duration(processing_time)
                audio_duration_str = self.format_duration(audio_duration)
                # Calculate speed
                speed = audio_duration / processing_time
                speed = round(speed, 2)
                speed_str = f"Speed: ~{speed}x"

                # Map avg_logprob to confidence level
                if hasattr(self, 'avg_logprob') and self.avg_logprob is not None:
                    avg_logprob = self.avg_logprob
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

                total_words = len(' '.join([s['text'] for s in self.segments]).split())
                total_segments = len(self.segments)

                efficiency_report = f"Processing time: {processing_time_str}\n" \
                                    f"Audio duration: {audio_duration_str}\n" \
                                    f"{speed_str}\n" \
                                    f"Confidence: {confidence}\n" \
                                    f"Total segments: {total_segments}\n" \
                                    f"Total words: {total_words}"

                self.queue.put("Efficiency Report:\n")
                self.queue.put(efficiency_report + "\n")

        except Exception as e:
            tb = traceback.format_exc()
            self.queue.put(f"Error: {e}\n{tb}")
            self.segments = None  # Clear segments on error
        finally:
            # Re-enable the start button
            self.queue.put(('ENABLE_START_BUTTON', None))

    def save_output(self):
        if not hasattr(self, 'segments') or not self.segments:
            messagebox.showwarning("Warning", "No transcription available to save.")
            return

        # Get the base name of the media file without extension
        media_filename = os.path.splitext(os.path.basename(self.file_entry.get()))[0]
        lang_selected = self.lang_var.get()
        language_code = language_name_to_code.get(lang_selected, "auto")

        if self.task_var.get() == "transcribe":
            default_srt_name = f"{media_filename}.{language_code}.srt"
        else:
            default_srt_name = f"{media_filename}.en.srt"

        # Ask the user to select the output directory
        directory = filedialog.askdirectory()
        if directory:
            file_path = os.path.join(directory, default_srt_name)
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    for i, segment in enumerate(self.segments, start=1):
                        # Format start and end times
                        start = self.format_timestamp(segment['start'])
                        end = self.format_timestamp(segment['end'])
                        # Write SRT block
                        f.write(f"{i}\n{start} --> {end}\n{segment['text'].strip()}\n\n")

                    # Add credits at the end
                    self.add_credits(f, i+1, segment['end'])
                messagebox.showinfo("Success", f"Subtitles saved")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save subtitles: {e}")

    def add_credits(self, file_handle, index, start_time):
        # Add credits as the final subtitle
        end_time = start_time + 5  # Display credits for 5 seconds
        start = self.format_timestamp(start_time + 1)  # Start 1 second after the last subtitle
        end = self.format_timestamp(end_time)
        task = self.task_var.get()
        if task.lower() == "transcribe":
            credit_text = "Transcribed by Veil's Transcription Service | Contact veilbreaker@voidmaster.xyz"
        else:
            credit_text = "Translated by Veil's Translation Service | Contact veilbreaker@voidmaster.xyz"
        file_handle.write(f"{index}\n{start} --> {end}\n{credit_text}\n\n")

    def format_timestamp(self, seconds):
        # Convert seconds to hours:minutes:seconds,milliseconds
        delta = datetime.timedelta(seconds=seconds)
        timestamp = (datetime.datetime(1,1,1) + delta).strftime("%H:%M:%S,%f")[:-3]
        return timestamp

    def format_duration(self, seconds):
        # Convert seconds to Hh Mm Ss format
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"

    def check_queue(self):
        try:
            while True:
                message = self.queue.get_nowait()
                if isinstance(message, tuple):
                    # Handle special commands
                    command, data = message
                    if command == 'ENABLE_START_BUTTON':
                        self.start_button.config(state=tk.NORMAL)
                else:
                    # Regular message
                    self.output_text.insert(tk.END, message)
        except queue.Empty:
            pass
        self.after(100, self.check_queue)  # Check the queue every 100 ms

if __name__ == "__main__":
    app = WhisperApp()
    app.mainloop()
