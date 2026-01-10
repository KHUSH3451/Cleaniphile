# 🎙️ Cleaniphile | AI Voice Enhancement System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-red) ![License](https://img.shields.io/badge/License-MIT-green)

**Cleaniphile** is an advanced AI-powered audio processing application designed to isolate human speech from complex environmental background noise. Using deep learning source separation (Demucs architecture), it provides studio-quality voice enhancement for educational and professional demonstration.

---

## 🎯 Problem Statement
Traditional noise filters (like spectral gating) often degrade voice quality, leaving behind "musical noise" or robotic artifacts. In scenarios with non-stationary noise (traffic, crowds, music), extracting clear speech is a significant challenge.

**Solution:** Cleaniphile leverages a **U-Net** based neural network trained on massive datasets to spectrally separation the "vocal" component from the audio mixture, preserving the natural intonation of the human voice.

## ⚙️ Architecture & Tech Stack

### 🛠️ Technology Stack
- **Languages**: Python 3.9+
- **Deep Learning Framework**: PyTorch, Torchaudio
- **Model Architecture**: Hybrid Demucs (Deep Music Separation)
- **User Interface**: Streamlit
- **Audio Processing**: Librosa, SoundFile, NumPy

### 🧠 How It Works
1.  **Input**: User uploads a `.wav` or `.mp3` file (resampled to 44.1kHz).
2.  **Preprocessing**: The audio signal is normalized and converted into a tensor.
3.  **Inference (Demucs)**:
    *   The model uses a Bi-LSTM encoder-decoder architecture.
    *   It decomposes the audio into four stems: *Vocals, Drums, Bass, Other*.
4.  **Extraction**: The system isolates the `Vocals` stem, discarding the noise components.
5.  **Output**: The cleaned vocal track is reconstructed and available for download.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Cleaniphile.git
cd Cleaniphile
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: PyTorch will be installed automatically. For GPU support, verify your CUDA version.*

### 3. Run the Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure
```bash
Cleaniphile/
├── app.py                # Main Streamlit Dashboard
├── requirements.txt      # Dependency Definitions
├── .gitignore           # Git Configuration
├── src/
│   ├── __init__.py
│   ├── model.py          # Demucs Model Wrapper (PyTorch)
│   ├── inference.py      # Audio Processing Pipeline
│   └── audio_utils.py    # Spectrogram & Waveform Utilities
└── README.md             # Documentation
```

## 📸 Usage
1.  Launch the app via `streamlit run app.py`.
2.  Drag and drop your noisy audio file (`.wav` or `.mp3`).
3.  Click **"Enhance Audio"**.
4.  Wait for the AI to process (First run may download model weights).
5.  Compare the **Original** vs **Enhanced** spectrograms and download the result.

## 🔮 Future Enhancements
- [ ] Real-time noise suppression for live mic input.
- [ ] Deployment to Hugging Face Spaces / Streamlit Cloud.
- [ ] Support for batch processing multiple files.
- [ ] Mobile-responsive optimization.

## 🤝 Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any features or bug fixes.

## 📜 License
This project is open-source and available under the **MIT License**.

---
*Developed for Academic Research & Demonstration.*
