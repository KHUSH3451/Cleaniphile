import streamlit as st
import os
import soundfile as sf
import time
from src.inference import VoiceCleaner
from src.audio_utils import plot_waveform, plot_spectrogram
import matplotlib.pyplot as plt
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Cleaniphile | AI Voice Enhancement",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #1e3d59;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #1e3d59;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff6e40;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff5722;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #1e3d59;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        color: #155724;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Singleton Model Loader ---
@st.cache_resource
def load_cleaner():
    return VoiceCleaner()

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/audio-wave.png", width=80)
        st.title("Cleaniphile")
        st.caption("v1.0.0 • Academic Release")
        st.markdown("---")
        
        st.subheader("🔧 System Architecture")
        st.markdown("""
        **Model:** Demucs (Hybrid Source Separation)  
        **Framework:** PyTorch & Torchaudio  
        **Backend:** Python 3.8+  
        """)
        
        st.markdown("---")
        st.info(
            "This application uses Deep Learning to isolate human speech "
            "from complex background environmental noise."
        )

    # --- Main Content ---
    st.title("🎙️ Cleaniphile")
    st.subheader("Professional AI Voice Enhancement System")
    st.markdown("Upload a noisy audio recording to enhance speech clarity using state-of-the-art source separation.")

    st.markdown("---")

    # 1. File Upload Section
    col_upload, col_empty = st.columns([1, 2]) # Keep upload compact? Or full width? Let's do full width usually better.
    # Actually, for wide layout, a centered approach looks nice, or simply full width.
    uploaded_file = st.file_uploader(
        "📂 Drag and drop or browse to upload an audio file", 
        type=["wav", "mp3"], 
        help="Supported formats: WAV (PCM), MP3. Max size: 200MB"
    )

    if uploaded_file is not None:
        # Determine file details
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext == "mp3":
            audio_format = "audio/mp3"
        else:
            audio_format = "audio/wav"

        # Temporary Save
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        input_path = os.path.join(temp_dir, f"input.{file_ext}")
        
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Inspect Audio
        y_in, sr_in = sf.read(input_path)
        duration = len(y_in) / sr_in

        # Display Input Details
        st.markdown("### 🎧 Input Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sample Rate", f"{sr_in} Hz")
        with col2:
            st.metric("Duration", f"{duration:.2f} s")
        with col3:
            st.metric("Channels", "Stereo" if len(y_in.shape) > 1 else "Mono")

        st.audio(uploaded_file, format=audio_format)

        # 2. Processing Section
        st.markdown("---")
        
        # Load Model
        with st.status("Initializing AI Engine...", expanded=True) as status:
            st.write("Loading Deep Neural Network weights...")
            cleaner = load_cleaner()
            status.update(label="AI Engine Ready", state="complete", expanded=False)

        col_action, col_info = st.columns([1, 2])
        
        with col_action:
            process_btn = st.button("✨ Enhance Audio", use_container_width=True)
        
        with col_info:
            st.caption("Processing may take a few seconds depending on file length and hardware acceleration (CPU/GPU).")

        if process_btn:
            output_path = os.path.join(temp_dir, "cleaned_output.wav")
            
            with st.spinner("🔄 Denoising audio signal..."):
                start_time = time.time()
                try:
                    # Run Inference
                    cleaned_tensor, sr_out = cleaner.clean_audio(input_path)
                    
                    # Convert to Numpy & Save
                    cleaned_np = cleaned_tensor.cpu().numpy().T
                    sf.write(output_path, cleaned_np, sr_out)
                    
                    elapsed_time = time.time() - start_time
                    
                    # 3. Results Section
                    st.success(f"Processing Complete! Time taken: {elapsed_time:.2f} seconds")
                    
                    st.markdown("### 📊 Enhancement Results")
                    
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.markdown("**Original Signal**")
                        st.audio(input_path)
                        with st.expander("View Original Spectrogram"):
                            st.pyplot(plot_spectrogram(y_in, sr_in, "Noisy Spectrogram"))

                    with res_col2:
                        st.markdown("**Enhanced Signal**")
                        st.audio(output_path)
                        with st.expander("View Enhanced Spectrogram"):
                            st.pyplot(plot_spectrogram(cleaned_np.T, sr_out, "Cleaned Spectrogram"))
                    
                    st.markdown("---")
                    
                    # Download
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Enhanced Audio (.wav)",
                            data=f,
                            file_name=f"cleaniphile_enhanced_{int(time.time())}.wav",
                            mime="audio/wav",
                            use_container_width=True
                        )
                        
                except Exception as e:
                    st.error(f"Processing Error: {str(e)}")
                    st.exception(e)

if __name__ == "__main__":
    main()
