# PolUVR 🎶

[![PyPI version](https://badge.fury.io/py/PolUVR.svg?icon=si%3Apython)](https://badge.fury.io/py/PolUVR)
[![Open In Huggingface](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/Politrees/PolUVR)

# Overview

**PolUVR** is a Python-based audio separation tool that leverages advanced machine learning models to separate audio tracks into distinct stems, such as vocals, instrumental, drums, bass, and more. Built as a fork of the [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator), PolUVR offers enhanced usability, hardware acceleration, and a user-friendly Gradio interface.

---

# Key Features

- **Audio Separation:** Extract vocals, instrumental, drums, bass, and other stems.
- **Hardware Acceleration:** Supports CUDA (Nvidia GPUs) and CoreML (Apple Silicon).
- **Cross-Platform:** Works on Linux, macOS, and Windows.
- **Gradio Interface:** Easy-to-use web interface for audio separation.

---

# Installation 🛠️

### Hardware Acceleration Options

PolUVR supports multiple hardware acceleration options for optimal performance. To verify successful configuration, run:
```sh
PolUVR --env_info
```

| **Command**                 | **Expected Log Message**                                                   |
|-----------------------------|----------------------------------------------------------------------------|
| `pip install "PolUVR[gpu]"` | `ONNXruntime has CUDAExecutionProvider available, enabling acceleration`   |
| `pip install "PolUVR[cpu]"` | `ONNXruntime has CoreMLExecutionProvider available, enabling acceleration` |
| `pip install "PolUVR[cpu]"` | No hardware acceleration enabled                                           |

---

### FFmpeg Dependency

PolUVR relies on FFmpeg for audio processing. To check if FFmpeg is installed, run:
```sh
PolUVR --env_info
```
The log should show: `FFmpeg installed`

If FFmpeg is missing, install it using the following commands:

| **OS**            | **Command**                                                                                                     |
|-------------------|-----------------------------------------------------------------------------------------------------------------|
| **Debian/Ubuntu** | `apt-get update; apt-get install -y ffmpeg`                                                                     |
| **macOS**         | `brew update; brew install ffmpeg`                                                                              |
| **Windows**       | Follow this guide: [Install FFmpeg on Windows](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) |

If you cloned the repository, you can install FFmpeg with:
```sh
PolUVR-ffmpeg
```

---

## GPU / CUDA Specific Installation Steps

While installing `PolUVR` with the `[gpu]` extra should suffice, sometimes PyTorch and ONNX Runtime with CUDA support require manual intervention. If you encounter issues, follow these steps:

```sh
pip uninstall torch onnxruntime
pip cache purge
pip install --force-reinstall torch torchvision torchaudio
pip install --force-reinstall onnxruntime-gpu
```

For the latest PyTorch version, use the command recommended by the [PyTorch installation wizard](https://pytorch.org/get-started/locally/).

### Multiple CUDA Library Versions

If you need to install multiple CUDA versions (e.g., CUDA 11 alongside CUDA 12), use:
```sh
apt update; apt install nvidia-cuda-toolkit
```

If you encounter errors like `Failed to load library` or `cannot open shared object file`, resolve them by running:
```sh
python -m pip install ort-nightly-gpu --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12-nightly/pypi/simple/
```

---

# Usage 🚀

## Gradio Interface

To launch the Gradio interface, use:
```sh
PolUVR-app [--share] [--open]
```

| **Parameter** | **Description**                                                                |
|---------------|--------------------------------------------------------------------------------|
| `--share`     | Opens public access to the interface (useful for servers, Google Colab, etc.). |
| `--open`      | Automatically opens the interface in a new browser tab.                        |

As soon as one of the following messages appears:
```
Running on local URL:  http://127.0.0.1:7860
```
```
Running on public URL: https://28425b3eb261b9ddc6.gradio.live
```
you can click on the link to open the WebUI.

### Integrate Our Interface into Your Gradio Projects  
You can seamlessly incorporate our pre-built interface into your Gradio applications by using the following import statement:

```python
from PolUVR.utils import PolUVR_UI

# This function can be integrated into any section of your interface
PolUVR_UI()
```

The `PolUVR_UI` function supports two optional parameters:
- **`default_model_file_dir`:** (Optional) Directory path for downloading models. `Default: /tmp/PolUVR-models/`
- **`default_output_dir`:** (Optional) Directory path for saving separation results. `Default: output`

The default parameter values are configured as follows:
```python
PolUVR_UI(default_model_file_dir="/tmp/PolUVR-models/", default_output_dir="output")
```

To customize storage locations for model files and output results according to your project structure, specify alternative default paths:
```python
PolUVR_UI("UVR_models", "separation_results")
```

---

## Command Line Interface (CLI)

You can use Audio Separator via the command line, for example:

```sh
PolUVR /path/to/your/input/audio.wav --model_filename UVR-MDX-NET-Inst_HQ_3.onnx
```

This command will download the specified model file, process the `audio.wav` input audio and generate two new files in the current directory, one containing vocals and one containing instrumental.

**Note:** You do not need to download any files yourself - PolUVR does that automatically for you!

### Full command-line interface options

```sh
usage: PolUVR [-h] [-v] [-d] [-e] [-l] [--log_level LOG_LEVEL] [--list_filter LIST_FILTER] [--list_limit LIST_LIMIT] [--list_format {pretty,json}] [-m MODEL_FILENAME] [--output_format OUTPUT_FORMAT]
                       [--output_bitrate OUTPUT_BITRATE] [--output_dir OUTPUT_DIR] [--model_file_dir MODEL_FILE_DIR] [--download_model_only] [--invert_spect] [--normalization NORMALIZATION]
                       [--amplification AMPLIFICATION] [--single_stem SINGLE_STEM] [--sample_rate SAMPLE_RATE] [--use_soundfile] [--use_autocast] [--custom_output_names CUSTOM_OUTPUT_NAMES]
                       [--mdx_segment_size MDX_SEGMENT_SIZE] [--mdx_overlap MDX_OVERLAP] [--mdx_batch_size MDX_BATCH_SIZE] [--mdx_hop_length MDX_HOP_LENGTH] [--mdx_enable_denoise] [--vr_batch_size VR_BATCH_SIZE]
                       [--vr_window_size VR_WINDOW_SIZE] [--vr_aggression VR_AGGRESSION] [--vr_enable_tta] [--vr_high_end_process] [--vr_enable_post_process]
                       [--vr_post_process_threshold VR_POST_PROCESS_THRESHOLD] [--demucs_segment_size DEMUCS_SEGMENT_SIZE] [--demucs_shifts DEMUCS_SHIFTS] [--demucs_overlap DEMUCS_OVERLAP]
                       [--demucs_segments_enabled DEMUCS_SEGMENTS_ENABLED] [--mdxc_segment_size MDXC_SEGMENT_SIZE] [--mdxc_override_model_segment_size] [--mdxc_overlap MDXC_OVERLAP]
                       [--mdxc_batch_size MDXC_BATCH_SIZE] [--mdxc_pitch_shift MDXC_PITCH_SHIFT]
                       [audio_files ...]

Separate audio file into different stems.

positional arguments:
  audio_files                                            The audio file paths or directory to separate, in any common format.

options:
  -h, --help                                             show this help message and exit

Info and Debugging:
  -v, --version                                          Show the program's version number and exit.
  -d, --debug                                            Enable debug logging, equivalent to --log_level=debug.
  -e, --env_info                                         Print environment information and exit.
  -l, --list_models                                      List all supported models and exit. Use --list_filter to filter/sort the list and --list_limit to show only top N results.
  --log_level LOG_LEVEL                                  Log level, e.g. info, debug, warning (default: info).
  --list_filter LIST_FILTER                              Filter and sort the model list by 'name', 'filename', or any stem e.g. vocals, instrumental, drums
  --list_limit LIST_LIMIT                                Limit the number of models shown
  --list_format {pretty,json}                            Format for listing models: 'pretty' for formatted output, 'json' for raw JSON dump

Separation I/O Params:
  -m MODEL_FILENAME, --model_filename MODEL_FILENAME     Model to use for separation (default: model_bs_roformer_ep_317_sdr_12.9755.yaml). Example: -m 2_HP-UVR.pth
  --output_format OUTPUT_FORMAT                          Output format for separated files, any common format (default: FLAC). Example: --output_format=MP3
  --output_bitrate OUTPUT_BITRATE                        Output bitrate for separated files, any ffmpeg-compatible bitrate (default: None). Example: --output_bitrate=320k
  --output_dir OUTPUT_DIR                                Directory to write output files (default: <current dir>). Example: --output_dir=/app/separated
  --model_file_dir MODEL_FILE_DIR                        Model files directory (default: /tmp/PolUVR-models/). Example: --model_file_dir=/app/models
  --download_model_only                                  Download a single model file only, without performing separation.

Common Separation Parameters:
  --invert_spect                                         Invert secondary stem using spectrogram (default: False). Example: --invert_spect
  --normalization NORMALIZATION                          Max peak amplitude to normalize input and output audio to (default: 0.9). Example: --normalization=0.7
  --amplification AMPLIFICATION                          Min peak amplitude to amplify input and output audio to (default: 0.0). Example: --amplification=0.4
  --single_stem SINGLE_STEM                              Output only single stem, e.g. Instrumental, Vocals, Drums, Bass, Guitar, Piano, Other. Example: --single_stem=Instrumental
  --sample_rate SAMPLE_RATE                              Modify the sample rate of the output audio (default: 44100). Example: --sample_rate=44100
  --use_soundfile                                        Use soundfile to write audio output (default: False). Example: --use_soundfile
  --use_autocast                                         Use PyTorch autocast for faster inference (default: False). Do not use for CPU inference. Example: --use_autocast
  --custom_output_names CUSTOM_OUTPUT_NAMES              Custom names for all output files in JSON format (default: None). Example: --custom_output_names='{"Vocals": "vocals_output", "Drums": "drums_output"}'

MDX Architecture Parameters:
  --mdx_segment_size MDX_SEGMENT_SIZE                    Larger consumes more resources, but may give better results (default: 256). Example: --mdx_segment_size=256
  --mdx_overlap MDX_OVERLAP                              Amount of overlap between prediction windows, 0.001-0.999. Higher is better but slower (default: 0.25). Example: --mdx_overlap=0.25
  --mdx_batch_size MDX_BATCH_SIZE                        Larger consumes more RAM but may process slightly faster (default: 1). Example: --mdx_batch_size=4
  --mdx_hop_length MDX_HOP_LENGTH                        Usually called stride in neural networks, only change if you know what you're doing (default: 1024). Example: --mdx_hop_length=1024
  --mdx_enable_denoise                                   Enable denoising during separation (default: False). Example: --mdx_enable_denoise

VR Architecture Parameters:
  --vr_batch_size VR_BATCH_SIZE                          Number of batches to process at a time. Higher = more RAM, slightly faster processing (default: 1). Example: --vr_batch_size=16
  --vr_window_size VR_WINDOW_SIZE                        Balance quality and speed. 1024 = fast but lower, 320 = slower but better quality. (default: 512). Example: --vr_window_size=320
  --vr_aggression VR_AGGRESSION                          Intensity of primary stem extraction, -100 - 100. Typically, 5 for vocals & instrumentals (default: 5). Example: --vr_aggression=2
  --vr_enable_tta                                        Enable Test-Time-Augmentation; slow but improves quality (default: False). Example: --vr_enable_tta
  --vr_high_end_process                                  Mirror the missing frequency range of the output (default: False). Example: --vr_high_end_process
  --vr_enable_post_process                               Identify leftover artifacts within vocal output; may improve separation for some songs (default: False). Example: --vr_enable_post_process
  --vr_post_process_threshold VR_POST_PROCESS_THRESHOLD  Threshold for post_process feature: 0.1-0.3 (default: 0.2). Example: --vr_post_process_threshold=0.1

Demucs Architecture Parameters:
  --demucs_segment_size DEMUCS_SEGMENT_SIZE              Size of segments into which the audio is split, 1-100. Higher = slower but better quality (default: Default). Example: --demucs_segment_size=256
  --demucs_shifts DEMUCS_SHIFTS                          Number of predictions with random shifts, higher = slower but better quality (default: 2). Example: --demucs_shifts=4
  --demucs_overlap DEMUCS_OVERLAP                        Overlap between prediction windows, 0.001-0.999. Higher = slower but better quality (default: 0.25). Example: --demucs_overlap=0.25
  --demucs_segments_enabled DEMUCS_SEGMENTS_ENABLED      Enable segment-wise processing (default: True). Example: --demucs_segments_enabled=False

MDXC Architecture Parameters:
  --mdxc_segment_size MDXC_SEGMENT_SIZE                  Larger consumes more resources, but may give better results (default: 256). Example: --mdxc_segment_size=256
  --mdxc_override_model_segment_size                     Override model default segment size instead of using the model default value. Example: --mdxc_override_model_segment_size
  --mdxc_overlap MDXC_OVERLAP                            Amount of overlap between prediction windows, 2-50. Higher is better but slower (default: 8). Example: --mdxc_overlap=8
  --mdxc_batch_size MDXC_BATCH_SIZE                      Larger consumes more RAM but may process slightly faster (default: 1). Example: --mdxc_batch_size=4
  --mdxc_pitch_shift MDXC_PITCH_SHIFT                    Shift audio pitch by a number of semitones while processing. May improve output for deep/high vocals. (default: 0). Example: --mdxc_pitch_shift=2
```

---

## As a Dependency in a Python Project

You can use Audio Separator in your own Python project. Here's a minimal example using the default two stem (Instrumental and Vocals) model:

```python
from PolUVR.separator import Separator

# Initialize the Separator class (with optional configuration properties, below)
separator = Separator()

# Load a machine learning model (if unspecified, defaults to 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt')
separator.load_model()

# Perform the separation on specific audio files without reloading the model
output_files = separator.separate('audio1.wav')

print(f"Separation complete! Output file(s): {' '.join(output_files)}")
```

### Batch processing and processing with multiple models

You can process multiple files without reloading the model to save time and memory.

You only need to load a model when choosing or changing models. See example below:

```python
from PolUVR.separator import Separator

# Initialize the Separator class (with optional configuration properties, below)
separator = Separator()

# Load a model
separator.load_model(model_filename='UVR-MDX-NET-Inst_HQ_3.onnx')

# Separate multiple audio files without reloading the model
output_files = separator.separate(['audio1.wav', 'audio2.wav', 'audio3.wav'])

# Load a different model
separator.load_model(model_filename='UVR_MDXNET_KARA_2.onnx')

# Separate the same files with the new model
output_files = separator.separate(['audio1.wav', 'audio2.wav', 'audio3.wav'])
```

You can also specify the path to a folder containing audio files instead of listing the full paths to each of them:
```python
from PolUVR.separator import Separator

# Initialize the Separator class (with optional configuration properties, below)
separator = Separator()

# Load a model
separator.load_model(model_filename='UVR-MDX-NET-Inst_HQ_3.onnx')

# Separate all audio files located in a folder
output_files = separator.separate('path/to/audio_directory')
```

## Parameters for the Separator class

- **`log_level`:** (Optional) Logging level, e.g., INFO, DEBUG, WARNING. `Default: logging.INFO`
- **`log_formatter`:** (Optional) The log format. Default: None, which falls back to '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
- **`model_file_dir`:** (Optional) Directory to cache model files in. `Default: /tmp/PolUVR-models/`
- **`output_dir`:** (Optional) Directory where the separated files will be saved. If not specified, uses the current directory.
- **`output_format`:** (Optional) Format to encode output files, any common format (WAV, MP3, FLAC, M4A, etc.). `Default: WAV`
- **`normalization_threshold`:** (Optional) The amount by which the amplitude of the output audio will be multiplied. `Default: 0.9`
- **`amplification_threshold`:** (Optional) The minimum amplitude level at which the waveform will be amplified. If the peak amplitude of the audio is below this threshold, the waveform will be scaled up to meet it. `Default: 0.0`
- **`output_single_stem`:** (Optional) Output only a single stem, such as 'Instrumental' and 'Vocals'. `Default: None`
- **`invert_using_spec`:** (Optional) Flag to invert using spectrogram. `Default: False`
- **`sample_rate`:** (Optional) Set the sample rate of the output audio. `Default: 44100`
- **`use_soundfile`:** (Optional) Use soundfile for output writing, can solve OOM issues, especially on longer audio.
- **`use_autocast`:** (Optional) Flag to use PyTorch autocast for faster inference. Do not use for CPU inference. `Default: False`
- **`mdx_params`:** (Optional) MDX Architecture Specific Attributes & Defaults. `Default: {"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": False}`
- **`vr_params`:** (Optional) VR Architecture Specific Attributes & Defaults. `Default: {"batch_size": 1, "window_size": 512, "aggression": 5, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False}`
- **`demucs_params`:** (Optional) Demucs Architecture Specific Attributes & Defaults. `Default: {"segment_size": "Default", "shifts": 2, "overlap": 0.25, "segments_enabled": True}`
- **`mdxc_params`:** (Optional) MDXC Architecture Specific Attributes & Defaults. `Default: {"segment_size": 256, "override_model_segment_size": False, "batch_size": 1, "overlap": 8, "pitch_shift": 0}`

---

## Requirements 📋

- Python >= 3.10
- Libraries: torch, onnx, onnxruntime, numpy, librosa, requests, six, tqdm, pydub

---

# Developing Locally

### Prerequisites

- Python 3.10 or newer
- Conda (recommended: [Miniforge](https://github.com/conda-forge/miniforge))

### Clone the Repository

```sh
git clone https://github.com/Bebra777228/PolUVR.git
cd PolUVR
```

### Create and Activate the Conda Environment

```sh
conda env create
conda activate PolUVR-dev
```

### Install Dependencies

```sh
poetry install
```

For extra dependencies, use:
```sh
poetry install --extras "cpu"
```
or
```sh
poetry install --extras "gpu"
```

Install FFmpeg:
```sh
PolUVR-ffmpeg
```

### Running the Gradio interface Locally

```sh
PolUVR-app --open
```

### Deactivate the Virtual Environment

```sh
conda deactivate
```

---

# Contributing 🤝

Contributions are welcome! Fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to add.

---

# Acknowledgments

This project is a fork of the original [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) repository. Special thanks to the contributors of the original project for their foundational work.
