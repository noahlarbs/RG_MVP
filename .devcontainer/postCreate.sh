#!/usr/bin/env bash
set -e
pip install --upgrade pip
pip install -r requirements.txt
sudo apt-get update && sudo apt-get install -y ffmpeg
echo 'export WHISPER_MODEL=small' >> ~/.bashrc
