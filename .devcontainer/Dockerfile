FROM mcr.microsoft.com/devcontainers/python:3.11


RUN apt-get update \
    && apt-get install -y ffmpeg tesseract-ocr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
