## VTS-GUI

VTS-GUI (*Veil's Transcription/Translation Service*) is a desktop application that leverages [***OpenAI's Whisper***](https://github.com/openai/whisper) to generate subtitles from audio or video files. It support both transcription (subtitles in the original language) and translation (english subtitles for foreign-language), all through a user-friendly graphical interface.
## Key Features
- **Transcription:** Automatically detect and transcribe audio/video files into text subtitles, preserving the original language.

- **Translation:** Convert foreign-language speech into *English* subtitles directly from the source audio.

- **Model Selection:** Choose from various *Whisper* models (tiny to large/turbo) depending on your system's resources and desired processing speed/accuracy.

- **Device Flexibility:** Run on CPU or GPU if CUDA support is available.

- **Efficiency Metrics:** After processing, the GUI displays efficiency reports, including processing speed and confidence estimates.
## How It Works

Under the hood, VTS-GUI uses [***OpenAI's Whisper***](https://github.com/openai/whisper) as the backend engine. *Whisper* is a powerful speech recognition and translation model that can handle a wide range of languages and generate high-quality transcripts and translations.

When you load an audio or video file, *VTS-GUI* separates the actual audio channel from the media and passes it to *Whisper*. Depending on your selected task:

- **Transcribe:** *Whisper* returns text segments in the audio's original language.

- **Translate:** *Whisper* directly produces English text segments from the foreign-language audio. Bypassing the need for a separate transcription step.

The resulting text segments are processed into an SRT format which can be saved after processing.
## Requirements

- **Python 3.9.9 or later:** Make sure you have a compatible version of *Python* installed.

- **OpenAI Whisper:** *Whisper* is the core engine for transcription and translation. It is automatically installed from *OpenAI's GitHub* repository when you install this package.

- **Torch:** Required by *Whisper*. Installing this package will pull in *Torch* automatically.

- **ffmpeg & ffprobe:** These tools are required to extract media metadata (e.g. audio/video duration).

    - **Windows:** The best way to satisfy these packages is to install them using *Chocolately*
        
        **1.** Install *Chocolately* by frist opening an Administrator PowerShell instance, and run the command below (from [***chocolately.org***]()https://chocolatey.org/install)
        
        `Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))`
        
        **2.** Install *ffmpeg* by running
            `choco install ffmpeg`

    - **Linux:** Use yur distribution's package manager (e.g. `sudo apt install ffmpeg ffprobe` on Debian/Ubuntu).

    **Note:** ffmpeg is an external tools and not Python packages, so they must be installed at the system level before running VTS-GUI.
## Installation

### Install via GitHub (pip)

You can install *VTS-GUI* directly from GitHub using *pip*:

`pip install git+https://github.com/menevia16a/VTS-GUI.git`

This command will clone the repository, install the package along with its dependencies (including *Whisper*), and set up the vts-gui script so you can easily run the GUI
## Running the Application

Once installed, simply run: 

`vts-gui`

This will launch the GUI. From there, you can select your media file, choose a model, configure settings, and start transcription or translation. After processing, subtitles can be saved in SRT format.
## Usage Tips

- **Choosing a Model:** If you're on a machine with limited resources, start with the tiny or base model. These require less RAM and VRAM and are faster to run.

- **Transcription vs Translation:**
    - **Transcribe:** Original language subtitles (e.g. *German* speech → *German* subtitles).
    - **Translate:** Direct *English* subtitles (e.g. *German* speech → *English* subtitles).

- **English-Only Models:** Some whisper models (tiny.en, base.en, etc.) are English-Only variants. Check the English Only box if you'd like to use these models.
## Contact

For inquiries, feedback, or support, please contact:
***veilbreaker@voidmaster.xyz***

We welcome suggestions and bug reports.