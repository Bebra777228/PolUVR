# PolUVR 🎶

[![PyPI version](https://badge.fury.io/py/PolUVR.svg)](https://badge.fury.io/py/PolUVR)

## Overview

PolUVR is a Python-based audio separation tool that leverages advanced machine learning models to separate audio tracks into different stems, such as vocals, instrumental, drums, bass, and more. This project is a fork of the [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) repository, and it aims to provide a user-friendly interface for audio separation tasks.

---

## Installation 🛠️

### Hardware Acceleration Options

#### Nvidia GPU with CUDA

**Supported CUDA Versions:** 11.8 and 12.2

To verify successful configuration, run `PolUVR --env_info`. You should see the following log message:
```
ONNXruntime has CUDAExecutionProvider available, enabling acceleration
```

**Installation:**
```sh
pip install "PolUVR[gpu]"
```

#### Apple Silicon, macOS Sonoma+ with M1 or newer CPU (CoreML acceleration)

To verify successful configuration, run `PolUVR --env_info`. You should see the following log message:
```
ONNXruntime has CoreMLExecutionProvider available, enabling acceleration
```

**Installation:**
```sh
pip install "PolUVR[cpu]"
```

#### CPU-Only (No Hardware Acceleration)

**Installation:**
```sh
pip install "PolUVR[cpu]"
```

---

### FFmpeg Dependency

To check if `PolUVR` is correctly configured to use FFmpeg, run `PolUVR --env_info`. The log should show:
```
FFmpeg installed
```

If you installed `PolUVR` using `conda` or `docker`, FFmpeg should already be available in your environment. Otherwise, install FFmpeg manually:

**Debian/Ubuntu:**
```sh
apt-get update; apt-get install -y ffmpeg
```

**macOS:**
```sh
brew update; brew install ffmpeg
```

---

## GPU / CUDA Specific Installation Steps

In theory, installing `PolUVR` with the `[gpu]` extra should suffice. However, sometimes PyTorch and ONNX Runtime with CUDA support can be tricky. You may need to reinstall these packages directly:

```sh
pip uninstall torch onnxruntime
pip cache purge
pip install --force-reinstall torch torchvision torchaudio
pip install --force-reinstall onnxruntime-gpu
```

For the latest PyTorch version, use the command recommended by the [PyTorch installation wizard](https://pytorch.org/get-started/locally/).

### Multiple CUDA Library Versions

Depending on your environment, you may need specific CUDA library versions. For example, Google Colab uses CUDA 12 by default, but ONNX Runtime may still require CUDA 11 libraries. Install CUDA 11 libraries alongside CUDA 12:

```sh
apt update; apt install nvidia-cuda-toolkit
```

If you encounter errors like `Failed to load library` or `cannot open shared object file`, resolve them by running:

```sh
python -m pip install ort-nightly-gpu --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12-nightly/pypi/simple/
```

---

## Usage 🚀

### Command Line Interface (CLI)

Separate an audio file using the default model:

```sh
PolUVR /path/to/your/input/audio.wav --model_filename UVR-MDX-NET-Inst_HQ_3.onnx
```

This command will download the specified model, process `audio.wav`, and generate two files: one for vocals and one for instrumental.

**List Supported Models:**
```sh
PolUVR --list_models
```

### Full Command-Line Interface Options

```sh
usage: PolUVR [-h] [-v] [-d] [-e] [-l] [--log_level LOG_LEVEL] [-m MODEL_FILENAME] [--output_format OUTPUT_FORMAT] [--output_dir OUTPUT_DIR] [--model_file_dir MODEL_FILE_DIR]
	      [--invert_spect] [--normalization NORMALIZATION] [--single_stem SINGLE_STEM] [--sample_rate SAMPLE_RATE] [--use_autocast] [--custom_output_names]
	      [--mdx_segment_size MDX_SEGMENT_SIZE] [--mdx_overlap MDX_OVERLAP] [--mdx_batch_size MDX_BATCH_SIZE] [--mdx_hop_length MDX_HOP_LENGTH] [--mdx_enable_denoise]
	      [--vr_batch_size VR_BATCH_SIZE] [--vr_window_size VR_WINDOW_SIZE] [--vr_aggression VR_AGGRESSION] [--vr_enable_tta] [--vr_high_end_process] [--vr_enable_post_process] [--vr_post_process_threshold VR_POST_PROCESS_THRESHOLD]
	      [--demucs_segment_size DEMUCS_SEGMENT_SIZE] [--demucs_shifts DEMUCS_SHIFTS] [--demucs_overlap DEMUCS_OVERLAP] [--demucs_segments_enabled DEMUCS_SEGMENTS_ENABLED]
	      [--mdxc_segment_size MDXC_SEGMENT_SIZE] [--mdxc_override_model_segment_size] [--mdxc_overlap MDXC_OVERLAP] [--mdxc_batch_size MDXC_BATCH_SIZE] [--mdxc_pitch_shift MDXC_PITCH_SHIFT]
              [audio_file]

Separate audio file into different stems.

positional arguments:
  audio_file                                             The audio file path to separate, in any common format.

options:
  -h, --help                                             Show this help message and exit.

Info and Debugging:
  -v, --version                                          Show the program version number and exit.
  -d, --debug                                            Enable debug logging, equivalent to --log_level=debug.
  -e, --env_info                                         Print environment information and exit.
  -l, --list_models                                      List all supported models and exit.
  --log_level LOG_LEVEL                                  Log level, e.g., info, debug, warning (default: info).

Separation I/O Params:
  -m MODEL_FILENAME, --model_filename MODEL_FILENAME     Model to use for separation (default: UVR-MDX-NET-Inst_HQ_3.onnx). Example: -m 2_HP-UVR.pth
  --output_format OUTPUT_FORMAT                          Output format for separated files, any common format (default: FLAC). Example: --output_format=MP3
  --output_dir OUTPUT_DIR                                Directory to write output files (default: <current dir>). Example: --output_dir=/app/separated
  --model_file_dir MODEL_FILE_DIR                        Model files directory (default: /tmp/PolUVR-models/). Example: --model_file_dir=/app/models

Common Separation Parameters:
  --invert_spect                                         Invert secondary stem using spectogram (default: False). Example: --invert_spect
  --normalization NORMALIZATION                          Value by which to multiply the amplitude of the output files (default: 0.9). Example: --normalization=0.7
  --single_stem SINGLE_STEM                              Output only single stem, e.g., Instrumental, Vocals, Drums, Bass, Guitar, Piano, Other. Example: --single_stem=Instrumental
  --sample_rate SAMPLE_RATE                              Set the sample rate of the output audio (default: 44100). Example: --sample_rate=44100
  --use_autocast                                         Use PyTorch autocast for faster inference (default: False). Do not use for CPU inference. Example: --use_autocast
  --custom_output_names                                  Custom names for all output files in JSON format (default: None). Example: --custom_output_names='{"Vocals": "vocals_output", "Drums": "drums_output"}'

MDX Architecture Parameters:
  --mdx_segment_size MDX_SEGMENT_SIZE                    Larger consumes more resources, but may give better results (default: 256). Example: --mdx_segment_size=256
  --mdx_overlap MDX_OVERLAP                              Amount of overlap between prediction windows, 0.001-0.999. Higher is better but slower (default: 0.25). Example: --mdx_overlap=0.25
  --mdx_batch_size MDX_BATCH_SIZE                        Larger consumes more RAM but may process slightly faster (default: 1). Example: --mdx_batch_size=4
  --mdx_hop_length MDX_HOP_LENGTH                        Usually called stride in neural networks; only change if you know what you do (default: 1024). Example: --mdx_hop_length=1024
  --mdx_enable_denoise                                   Enable denoising after separation (default: False). Example: --mdx_enable_denoise

VR Architecture Parameters:
  --vr_batch_size VR_BATCH_SIZE                          Number of "batches" to process at a time. Higher = more RAM, slightly faster processing (default: 1). Example: --vr_batch_size=16
  --vr_window_size VR_WINDOW_SIZE                        Balance quality and speed. 1024 = fast but lower, 320 = slower but better quality (default: 512). Example: --vr_window_size=320
  --vr_aggression VR_AGGRESSION                          Intensity of primary stem extraction, -100 - 100. Typically 5 for vocals & instrumentals (default: 5). Example: --vr_aggression=2
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
  --mdxc_pitch_shift MDXC_PITCH_SHIFT                    Shift audio pitch by a number of semitones while processing. May improve output for deep/high vocals (default: 0). Example: --mdxc_pitch_shift=2
```

---

### As a Dependency in a Python Project

Use PolUVR in your Python project with the following example:

```python
from PolUVR.separator import Separator

# Initialize the Separator class
separator = Separator()

# Load a machine learning model
separator.load_model()

# Perform separation on specific audio files
output_files = separator.separate('audio1.wav')

print(f"Separation complete! Output file(s): {' '.join(output_files)}")
```

#### Batch Processing and Multiple Models

Process multiple files without reloading the model:

```python
from PolUVR.separator import Separator

separator = Separator()
separator.load_model(model_filename='UVR-MDX-NET-Inst_HQ_3.onnx')

output_file_paths_1 = separator.separate('audio1.wav')
output_file_paths_2 = separator.separate('audio2.wav')
output_file_paths_3 = separator.separate('audio3.wav')

separator.load_model(model_filename='UVR_MDXNET_KARA_2.onnx')

output_file_paths_4 = separator.separate('audio1.wav')
output_file_paths_5 = separator.separate('audio2.wav')
output_file_paths_6 = separator.separate('audio3.wav')
```

#### Renaming Stems

You can rename the output files by specifying the desired names. For example:

```python
output_names = {
    "Vocals": "vocals_output",
    "Instrumental": "instrumental_output",
}
output_files = separator.separate('audio1.wav', output_names)
```

In this case, the output file names will be: `vocals_output.wav` and `instrumental_output.wav`.

You can also rename specific stems:

- To rename the Vocals stem:
  ```python
  output_names = {
      "Vocals": "vocals_output",
  }
  output_files = separator.separate('audio1.wav', output_names)
  ```
  > The output files will be named: `vocals_output.wav` and `audio1_(Instrumental)_model_mel_band_roformer_ep_3005_sdr_11.wav`

- To rename the Instrumental stem:
  ```python
  output_names = {
      "Instrumental": "instrumental_output",
  }
  output_files = separator.separate('audio1.wav', output_names)
  ```
  > The output files will be named: `audio1_(Vocals)_model_mel_band_roformer_ep_3005_sdr_11.wav` and `instrumental_output.wav`

- List of stems for Demucs models:
  - htdemucs_6s.yaml
    ```python
    output_names = {
        "Vocals": "vocals_output",
        "Drums": "drums_output",
        "Bass": "bass_output",
        "Other": "other_output",
        "Guitar": "guitar_output",
        "Piano": "piano_output",
    }
    ```
  - Other Demucs models
    ```python
    output_names = {
        "Vocals": "vocals_output",
        "Drums": "drums_output",
        "Bass": "bass_output",
        "Other": "other_output",
    }
    ```

## Parameters for the Separator Class

- **`log_level`:** (Optional) Logging level, e.g., INFO, DEBUG, WARNING. `Default: logging.INFO`
- **`log_formatter`:** (Optional) The log format. Default: None, which falls back to '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
- **`model_file_dir`:** (Optional) Directory to cache model files in. `Default: /tmp/PolUVR-models/`
- **`output_dir`:** (Optional) Directory where the separated files will be saved. If not specified, uses the current directory.
- **`output_format`:** (Optional) Format to encode output files, any common format (WAV, MP3, FLAC, M4A, etc.). `Default: WAV`
- **`normalization_threshold`:** (Optional) The amount by which the amplitude of the output audio will be multiplied. `Default: 0.9`
- **`amplification_threshold`:** (Optional) The minimum amplitude level at which the waveform will be amplified. If the peak amplitude of the audio is below this threshold, the waveform will be scaled up to meet it. `Default: 0.6`
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

## Developing Locally

### Prerequisites

- Python 3.10 or newer
- Conda (recommended: Miniforge)

### Clone the Repository

```sh
git clone https://github.com/YOUR_USERNAME/PolUVR.git
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

Install extra dependencies:
```sh
poetry install --extras "cpu"
```
or
```sh
poetry install --extras "gpu"
```

### Running the CLI Locally

```sh
PolUVR path/to/your/audio-file.wav
```

### Deactivate the Virtual Environment

```sh
conda deactivate
```

---

## Contributing 🤝

Contributions are welcome! Fork the repository, make your changes, and submit a pull request.

## Credits 🙏

- [Anjok07](https://github.com/Anjok07) - Author of [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui)
- [DilanBoskan](https://github.com/DilanBoskan)
- [Kuielab & Woosung Choi](https://github.com/kuielab)
- [KimberleyJSN](https://github.com/KimberleyJensen)
- [Hv](https://github.com/NaJeongMo/Colab-for-MDX_B)
- [zhzhongshi](https://github.com/zhzhongshi)

---

## Original Repository

This project is a fork of the original [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) repository.
