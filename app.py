import os, sys
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass


import streamlit as st
import json, tempfile
from pipeline import process_youtube, process_video_file

st.set_page_config(page_title="Responsible Gaming Shorts MVP", layout="wide")
st.title("Responsible Gaming Shorts — Classifier MVP")

mode = st.radio("Input type", ["YouTube URL", "Upload video (MP4)"])
source = None

if mode == "YouTube URL":
    url = st.text_input("YouTube video URL (Shorts preferred)")
    if st.button("Analyze") and url:
        with st.spinner("Downloading and analyzing..."):
            try:
                result = process_youtube(url)
                st.success("Done.")
                st.subheader(f"Overall risk score: {result['overall']}")
                st.json(result["categories"])
                st.write("Flags:")
                for cat, name in result["flags"]:
                    st.write(f"- **{cat}**: {name}")
                st.expander("Transcript").write(result["transcript"])
                st.expander("On-screen OCR text").write(result["ocr_text"])
                st.write("Representative frames:")
                cols = st.columns(3)
                for i, f in enumerate(result["rep_frames"]):
                    if i < len(cols):
                        cols[i].image(f, use_column_width=True)
            except Exception as e:
                st.error(str(e))

else:
    up = st.file_uploader("Upload a short video (.mp4)", type=["mp4"])
    if up is not None and st.button("Analyze upload"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(up.read())
            tmp_path = tmp.name
        with st.spinner("Analyzing..."):
            try:
                result = process_video_file(tmp_path)
                st.success("Done.")
                st.subheader(f"Overall risk score: {result['overall']}")
                st.json(result["categories"])
                st.write("Flags:")
                for cat, name in result["flags"]:
                    st.write(f"- **{cat}**: {name}")
                st.expander("Transcript").write(result["transcript"])
                st.expander("On-screen OCR text").write(result["ocr_text"])
                st.write("Representative frames:")
                cols = st.columns(3)
                for i, f in enumerate(result["rep_frames"]):
                    if i < len(cols):
                        cols[i].image(f, use_column_width=True)
            except Exception as e:
                st.error(str(e))
        os.unlink(tmp_path)
