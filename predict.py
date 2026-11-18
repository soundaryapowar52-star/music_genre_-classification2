"""
predict.py
----------
Audio Genre Prediction & Inference Script.

Predicts music genre from any audio file (.wav, .mp3, .ogg, .flac)
with exact confidence scores and optional spectrogram visualization.

Key Fixes:
1. Strict shape normalization: ensures all extracted segments match model input (130, 40, 1).
2. Works on short (<3s) and long tracks seamlessly.
3. Provides a clean ASCII confidence score breakdown.
4. Generates an optional 4-panel acoustic report (--plot).
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf

# Model & Feature Hyperparameters
MODEL_PATH = "models/music_genre_cnn.keras"
MAPPING_PATH = "models/genre_mapping.json"
SAMPLE_RATE = 22050
SEGMENT_DURATION = 3  # seconds
SAMPLES_PER_SEGMENT = int(SAMPLE_RATE * SEGMENT_DURATION)  # 66150
EXPECTED_FRAMES = 130
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512

DEFAULT_MAPPING = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]


def load_model_and_mapping(model_path=MODEL_PATH, mapping_path=MAPPING_PATH):
    """Loads the trained Keras CNN model and genre label list."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'!\n"
            "Please run 'python train.py' to train and save the model."
        )

    model = tf.keras.models.load_model(model_path)

    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as fp:
            mapping = json.load(fp)
    else:
        mapping = DEFAULT_MAPPING

    return model, mapping


def extract_segments(file_path):
    """
    Loads audio file and slices it into 3-second segments.
    Ensures every segment is strictly of shape (EXPECTED_FRAMES, N_MFCC).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file '{file_path}' does not exist.")

    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        raise RuntimeError(f"Error loading audio file '{file_path}': {e}")

    total_duration = len(signal) / sr
    num_segments = int(len(signal) / SAMPLES_PER_SEGMENT)

    # If audio is shorter than 3s, pad it to 3s
    if num_segments == 0:
        signal = np.pad(signal, (0, SAMPLES_PER_SEGMENT - len(signal)), mode="constant")
        num_segments = 1

    segments_mfcc = []

    for s in range(num_segments):
        start = s * SAMPLES_PER_SEGMENT
        finish = start + SAMPLES_PER_SEGMENT
        chunk = signal[start:finish]

        if len(chunk) < N_FFT:
            continue

        mfcc = librosa.feature.mfcc(
            y=chunk,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        ).T

        # Normalize shape strictly to (130, 40)
        if mfcc.shape[0] < EXPECTED_FRAMES:
            pad_len = EXPECTED_FRAMES - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_len), (0, 0)), mode="constant")
        elif mfcc.shape[0] > EXPECTED_FRAMES:
            mfcc = mfcc[:EXPECTED_FRAMES, :]

        segments_mfcc.append(mfcc)

    if len(segments_mfcc) == 0:
        raise ValueError("Could not extract any valid audio segments.")

    X = np.array(segments_mfcc, dtype=np.float32)[..., np.newaxis]
    return signal, sr, X, total_duration


def predict_audio_genre(audio_path, model, mapping):
    """Performs inference and averages probabilities across all audio segments."""
    signal, sr, X, duration = extract_segments(audio_path)

    # Predict probabilities for each 3-second window
    segment_predictions = model.predict(X, verbose=0)

    # Average probabilities across all windows (temporal ensemble)
    avg_probs = np.mean(segment_predictions, axis=0)
    winner_idx = int(np.argmax(avg_probs))
    winner_genre = mapping[winner_idx]
    confidence = float(avg_probs[winner_idx]) * 100.0

    return winner_genre, confidence, avg_probs, signal, sr, duration


def plot_acoustic_report(signal, sr, mapping, avg_probs, winner_genre, confidence, output_path="prediction_report.png"):
    """Generates a 4-panel visual report: Waveform, Mel-Spectrogram, MFCC, and Confidence Bar Chart."""
    fig = plt.figure(figsize=(16, 10))

    # 1. Waveform
    ax1 = fig.add_subplot(2, 2, 1)
    librosa.display.waveshow(signal, sr=sr, ax=ax1, color="#2563eb", alpha=0.85)
    ax1.set_title("1. Audio Waveform (Amplitude vs Time)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    # 2. Mel-Spectrogram
    ax2 = fig.add_subplot(2, 2, 2)
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img2 = librosa.display.specshow(mel_db, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel", ax=ax2, cmap="magma")
    fig.colorbar(img2, ax=ax2, format="%+2.0f dB")
    ax2.set_title("2. Mel-Frequency Spectrogram (dB)", fontsize=13, fontweight="bold")

    # 3. MFCC Heatmap
    ax3 = fig.add_subplot(2, 2, 3)
    mfcc_full = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    img3 = librosa.display.specshow(mfcc_full, sr=sr, hop_length=HOP_LENGTH, x_axis="time", ax=ax3, cmap="coolwarm")
    fig.colorbar(img3, ax=ax3)
    ax3.set_title(f"3. MFCC Features ({N_MFCC} Coefficients)", fontsize=13, fontweight="bold")
    ax3.set_ylabel("MFCC Index")

    # 4. Confidence Bar Chart
    ax4 = fig.add_subplot(2, 2, 4)
    colors = ["#22c55e" if g == winner_genre else "#94a3b8" for g in mapping]
    y_pos = np.arange(len(mapping))
    bars = ax4.barh(y_pos, avg_probs * 100, color=colors, edgecolor="#334155")
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(mapping, fontsize=11)
    ax4.invert_yaxis()
    ax4.set_xlabel("Confidence Score (%)", fontsize=12)
    ax4.set_xlim(0, 100)
    ax4.set_title(f"4. Confidence Scores (Winner: {winner_genre.upper()} - {confidence:.1f}%)", fontsize=13, fontweight="bold")
    ax4.grid(True, axis="x", alpha=0.3)

    for bar in bars:
        w = bar.get_width()
        if w > 3:
            ax4.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%",
                     va="center", ha="left", fontsize=9, fontweight="bold", color="#1e293b")

    plt.suptitle("Music Genre Classification - Prediction Report", fontsize=16, fontweight="heavy", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Visual report saved to '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Predict music genre from an audio file using Keras CNN.")
    parser.add_argument("audio_file", type=str, help="Path to input audio file (.wav, .mp3, etc.)")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to trained .keras model")
    parser.add_argument("--plot", action="store_true", help="Generate and save visual acoustic report")
    parser.add_argument("--output-plot", type=str, default="prediction_report.png", help="Path to save output plot")

    args = parser.parse_args()

    print("=" * 65)
    print("MUSIC GENRE CLASSIFICATION - PREDICTION")
    print("=" * 65)
    print(f"Audio file: {args.audio_file}")

    try:
        model, mapping = load_model_and_mapping(args.model)
        winner_genre, confidence, avg_probs, signal, sr, duration = predict_audio_genre(args.audio_file, model, mapping)
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        return

    print("\n" + "-" * 65)
    print(f"PREDICTED GENRE :  {winner_genre.upper()}")
    print(f"CONFIDENCE SCORE:  {confidence:.2f}%")
    print(f"AUDIO DURATION  :  {duration:.2f} seconds ({len(signal)} samples @ {sr}Hz)")
    print("-" * 65)

    print("\nGenre Probability Breakdown:")
    sorted_indices = np.argsort(avg_probs)[::-1]
    for rank, idx in enumerate(sorted_indices, 1):
        g_name = mapping[idx]
        prob = avg_probs[idx] * 100
        bars = "█" * int(prob / 3.33) + "░" * (30 - int(prob / 3.33))
        print(f"  {rank:2d}. {g_name:<12} [{bars}] {prob:6.2f}%")

    print("=" * 65)

    if args.plot:
        plot_acoustic_report(signal, sr, mapping, avg_probs, winner_genre, confidence, args.output_plot)


if __name__ == "__main__":
    main()
