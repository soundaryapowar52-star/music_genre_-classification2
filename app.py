"""
app.py
------
Interactive Streamlit Web Application for Music Genre Classification.

Features & Fixes:
1. Windows-safe temporary file handling (avoids WinError 32 permission lock).
2. Offline-safe styling and icons.
3. Audio playback widget for WAV, MP3, OGG, FLAC.
4. Real-time Waveform, Mel-Spectrogram, and MFCC visualizations.
5. Live CNN inference with confidence gauges and probability distribution.
"""

import os
import io
import json
import math
import tempfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf

# Streamlit Configuration
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "models/music_genre_cnn.keras"
MAPPING_PATH = "models/genre_mapping.json"
SAMPLE_RATE = 22050
SEGMENT_DURATION = 3  # seconds
SAMPLES_PER_SEGMENT = int(SAMPLE_RATE * SEGMENT_DURATION)  # 66150
EXPECTED_FRAMES = 130
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512

DEFAULT_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

GENRE_ICONS = {
    "blues": "🎷 Blues",
    "classical": "🎻 Classical",
    "country": "🤠 Country",
    "disco": "🪩 Disco",
    "hiphop": "🎤 Hip-Hop",
    "jazz": "🎺 Jazz",
    "metal": "⚡ Metal",
    "pop": "✨ Pop",
    "reggae": "🌴 Reggae",
    "rock": "🎸 Rock"
}


@st.cache_resource
def load_trained_model():
    """Loads and caches the trained Keras CNN model."""
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None


@st.cache_data
def load_mapping():
    """Loads genre mapping from JSON file or returns defaults."""
    if os.path.exists(MAPPING_PATH):
        try:
            with open(MAPPING_PATH, "r") as fp:
                return json.load(fp)
        except Exception:
            return DEFAULT_GENRES
    return DEFAULT_GENRES


def process_audio_upload(uploaded_bytes, file_ext):
    """
    Safely writes audio bytes to a temporary file on Windows,
    loads audio via librosa, and extracts 3-second MFCC slices.
    """
    tmp_path = None
    try:
        # Create temp file and close handle before librosa opens it (crucial for Windows!)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            tmp.write(uploaded_bytes)
            tmp_path = tmp.name

        signal, sr = librosa.load(tmp_path, sr=SAMPLE_RATE)
        duration = len(signal) / sr

        num_segments = int(len(signal) / SAMPLES_PER_SEGMENT)
        if num_segments == 0:
            signal_padded = np.pad(signal, (0, SAMPLES_PER_SEGMENT - len(signal)), mode="constant")
            num_segments = 1
        else:
            signal_padded = signal

        segments_mfcc = []
        for s in range(num_segments):
            start = s * SAMPLES_PER_SEGMENT
            finish = start + SAMPLES_PER_SEGMENT
            chunk = signal_padded[start:finish]

            if len(chunk) < N_FFT:
                continue

            mfcc = librosa.feature.mfcc(
                y=chunk, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
            ).T

            # Strict shape normalization to (130, 40)
            if mfcc.shape[0] < EXPECTED_FRAMES:
                pad_len = EXPECTED_FRAMES - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_len), (0, 0)), mode="constant")
            elif mfcc.shape[0] > EXPECTED_FRAMES:
                mfcc = mfcc[:EXPECTED_FRAMES, :]

            segments_mfcc.append(mfcc)

        X = np.array(segments_mfcc, dtype=np.float32)
        if len(X) > 0:
            X = X[..., np.newaxis]

        return signal, sr, duration, X, None

    except Exception as e:
        return None, None, None, None, str(e)
    finally:
        # Cleanup temporary file safely
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# Sidebar
with st.sidebar:
    st.markdown("## 🎵 Audio AI Classifier")
    st.markdown("""
    Predict music genre from any audio file using Deep Learning!
    
    * **Model:** 2D Convolutional Neural Network (CNN)
    * **Features:** 40-Coefficient MFCC + Mel-Spectrogram
    * **Windowing:** 3-second Temporal Slicing
    * **Dataset:** GTZAN (10 Genres)
    """)
    st.markdown("---")

    model = load_trained_model()
    mapping = load_mapping()

    if model is not None:
        st.success("Trained CNN Model Loaded! ✅")
    else:
        st.warning("Model file not found. Train the model using `python train.py` to enable live predictions!")


# Main Application
st.title("🎶 Music Genre Classification & Acoustic Analysis")
st.write("Upload a song or audio track to classify its genre, calculate confidence scores, and visualize acoustic frequency spectrograms.")

uploaded_file = st.file_uploader(
    "Choose an audio file (.wav, .mp3, .ogg, .flac):",
    type=["wav", "mp3", "ogg", "flac"]
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_ext = uploaded_file.name.split(".")[-1].lower()

    col_audio, col_meta = st.columns([1, 2])

    with col_audio:
        st.subheader("🎧 Audio Player")
        st.audio(file_bytes, format=f"audio/{file_ext}")
        st.caption(f"File: `{uploaded_file.name}` ({len(file_bytes) / 1024:.1f} KB)")

    with st.spinner("Extracting acoustic features and spectrograms..."):
        signal, sr, duration, X, error_msg = process_audio_upload(file_bytes, file_ext)

    if error_msg:
        st.error(f"Failed to process audio file: {error_msg}")
    else:
        with col_meta:
            st.markdown(f"**Track Duration:** `{duration:.2f} s` | **Sample Rate:** `{sr} Hz` | **Windows Extracted:** `{len(X)}`")

            # Inference
            if model is not None and len(X) > 0:
                predictions = model.predict(X, verbose=0)
                avg_probs = np.mean(predictions, axis=0)
                winner_idx = int(np.argmax(avg_probs))
                winner_genre = mapping[winner_idx]
                confidence = float(avg_probs[winner_idx]) * 100.0

                st.markdown("### 🏆 Prediction Results")
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    display_name = GENRE_ICONS.get(winner_genre, winner_genre.capitalize())
                    st.metric(label="Predicted Genre", value=display_name)
                with metric_col2:
                    st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

                st.progress(min(1.0, float(avg_probs[winner_idx])))
            else:
                st.info("💡 Model not trained yet. Run `python train.py` to enable live genre classification!")
                avg_probs = None
                winner_genre = None

        st.markdown("---")

        # Visualizations
        st.subheader("📊 Acoustic Visualizations")
        tab1, tab2, tab3, tab4 = st.tabs([
            "🌊 Mel-Spectrogram (dB)",
            "📈 Time-Domain Waveform",
            "🎛️ MFCC Heatmap",
            "📊 Genre Probabilities"
        ])

        with tab1:
            st.markdown("#### Mel-Frequency Spectrogram (Decibels)")
            fig, ax = plt.subplots(figsize=(10, 4))
            mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            img = librosa.display.specshow(mel_db, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set_title(f"Mel-Spectrogram - {uploaded_file.name}")
            st.pyplot(fig)
            plt.close()

        with tab2:
            st.markdown("#### Waveform (Amplitude vs Time)")
            fig, ax = plt.subplots(figsize=(10, 3.5))
            librosa.display.waveshow(signal, sr=sr, ax=ax, color="#2563eb", alpha=0.85)
            ax.set_title("Amplitude vs Time")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with tab3:
            st.markdown(f"#### MFCC Matrix ({N_MFCC} Coefficients)")
            fig, ax = plt.subplots(figsize=(10, 4))
            mfcc_full = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
            img = librosa.display.specshow(mfcc_full, sr=sr, hop_length=HOP_LENGTH, x_axis="time", ax=ax, cmap="coolwarm")
            fig.colorbar(img, ax=ax)
            ax.set_title("MFCC Features across Time")
            ax.set_ylabel("Coefficient Index")
            st.pyplot(fig)
            plt.close()

        with tab4:
            if avg_probs is not None:
                st.markdown("#### Genre Probability Rankings")
                sorted_idx = np.argsort(avg_probs)[::-1]
                cols = st.columns(2)
                for i, idx in enumerate(sorted_idx):
                    col = cols[i % 2]
                    g_name = mapping[idx]
                    p = float(avg_probs[idx]) * 100.0
                    with col:
                        icon_title = GENRE_ICONS.get(g_name, g_name.capitalize())
                        st.write(f"**{i+1}. {icon_title}**: `{p:.2f}%`")
                        st.progress(min(1.0, float(avg_probs[idx])))
            else:
                st.info("Train the model with `python train.py` to view probability distributions.")
else:
    st.info("👆 Upload an audio file above to test the classifier and visualize its spectrogram!")
