import streamlit as st
from speechbrain.pretrained import EncoderDecoderASR


# Classes
class ASR:
    def __init__(self):
        self.asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-en",
                                                        savedir="pretrained_models/asr-wav2vec2-commonvoice-en")

    def transcribe(self, audio_file):
        predicted_words = self.asr_model.transcribe_file(audio_file)
        return predicted_words


# Functions
@st.cache
def load_model():
    return ASR()

# App


"""
# Automatic Speech Recognition
"""
model = load_model()
audio_file = st.selectbox('Select Audio File: ', ['Choose a File', 'Sample1', 'Sample2', 'Sample3', 'Sample4', 'Sample5'])
if audio_file != 'Choose a File':
    """
    ### Audio File
    """
    st.audio(f"{audio_file}.wav")
    """
    ### Transcription
    """
    with st.spinner("Transcribing... Please wait..."):
        result = model.transcribe(f'{audio_file}.wav')
        st.write(result)
