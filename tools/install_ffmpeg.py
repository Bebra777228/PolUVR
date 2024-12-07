import os
import platform
import subprocess
import urllib.request


def install_ffmpeg():
    system = platform.system()

    if system == "Darwin":
        # macOS
        subprocess.run(["brew", "update"])
        subprocess.run(["brew", "install", "ffmpeg"])
    elif system == "Linux":
        # Linux
        subprocess.run(["sudo", "apt-get", "update"])
        subprocess.run(["sudo", "apt-get", "install", "-y", "ffmpeg"])
    elif system == "Windows":
        # Windows
        urls = [
            "https://huggingface.co/Politrees/RVC_resources/resolve/main/tools/ffmpeg/ffmpeg.exe?download=true",
            "https://huggingface.co/Politrees/RVC_resources/resolve/main/tools/ffmpeg/ffplay.exe?download=true",
            "https://huggingface.co/Politrees/RVC_resources/resolve/main/tools/ffmpeg/ffprobe.exe?download=true",
        ]
        for url in urls:
            filename = os.path.basename(url.split("?")[0])
            urllib.request.urlretrieve(url, filename)
    else:
        print(f"Unsupported OS: {system}")


def main():
    install_ffmpeg()


if __name__ == "__main__":
    main()
