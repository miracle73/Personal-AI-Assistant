import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import io
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Page Configuration
st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🤖",
    layout="wide"
)

# Load AI Model
@st.cache_resource
def load_model():
    return OllamaLLM(model="mistral")  # Change to "llama3" or another Ollama model

llm = load_model()

# Initialize Memory (LangChain v1.0+)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Initialize Speech Recognition
recognizer = sr.Recognizer()

# Function to Convert Text to Audio (returns audio bytes)
def text_to_audio(text, lang='en'):
    """Convert text to audio and return audio bytes"""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.getvalue()
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

# Define AI Chat Prompt
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Previous conversation: {chat_history}
User: {question}
AI: Please provide a helpful and concise response."""
)

# Function to Process AI Responses
def run_chain(question):
    """Process user question and generate AI response"""
    # Retrieve past chat history (limit to last 10 messages for context)
    messages = st.session_state.chat_history.messages[-10:] if len(st.session_state.chat_history.messages) > 10 else st.session_state.chat_history.messages
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in messages])
    
    # Run AI response generation
    with st.spinner("🤔 Thinking..."):
        response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))
    
    # Store new user input and AI response in memory
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)
    
    return response

# Function to process audio input
def process_audio_input(audio_file):
    """Process uploaded audio and return recognized text"""
    # Save the audio file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getvalue())
    
    # Process the audio file with speech recognition
    r = sr.Recognizer()
    try:
        with sr.AudioFile("temp_audio.wav") as source:
            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.record(source)
        
        # Try Google Speech Recognition
        user_query = r.recognize_google(audio)
        return user_query
    
    except sr.UnknownValueError:
        st.error("🤖 Sorry, I couldn't understand the audio. Please try again with clearer speech.")
        return None
    except sr.RequestError as e:
        st.error(f"⚠️ Speech Recognition Service Error: {e}")
        st.info("💡 Tip: Check your internet connection for Google Speech Recognition")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

# Main UI Layout
st.title("🤖 AI Voice Assistant")
st.markdown("---")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat Interface")
    
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["🎤 Voice Input", "⌨️ Text Input"])
    
    with tab1:
        st.write("Click the button below to record your voice:")
        
        # Audio Input
        audio_file = st.audio_input("Record your voice", key="audio_recorder")
        
        if audio_file:
            # Process the audio
            user_query = process_audio_input(audio_file)
            
            if user_query:
                st.success(f"✅ Recognized: **{user_query}**")
                
                # Process the query and get AI response
                ai_response = run_chain(user_query)
                
                # Display the conversation in a nice format
                st.markdown("### 🗣️ Conversation")
                
                # User message
                with st.chat_message("user"):
                    st.write(user_query)
                
                # AI message
                with st.chat_message("assistant"):
                    st.write(ai_response)
                
                # Convert AI response to audio
                st.markdown("### 🔊 Audio Response")
                audio_bytes = text_to_audio(ai_response)
                
                if audio_bytes:
                    # Create columns for audio player and download button
                    audio_col1, audio_col2 = st.columns([3, 1])
                    
                    with audio_col1:
                        st.audio(audio_bytes, format='audio/mp3', autoplay=True)
                    
                    with audio_col2:
                        st.download_button(
                            label="📥 Download",
                            data=audio_bytes,
                            file_name="ai_response.mp3",
                            mime="audio/mp3"
                        )
    
    with tab2:
        st.write("Type your message below:")
        
        # Text input with form to handle Enter key
        with st.form(key="text_input_form", clear_on_submit=True):
            text_input = st.text_input("Your message:", placeholder="Type your message here...")
            submit_button = st.form_submit_button("Send 📤")
        
        if submit_button and text_input:
            # Process the text input
            ai_response = run_chain(text_input)
            
            # Display the conversation
            st.markdown("### 🗣️ Conversation")
            
            # User message
            with st.chat_message("user"):
                st.write(text_input)
            
            # AI message
            with st.chat_message("assistant"):
                st.write(ai_response)
            
            # Convert AI response to audio
            st.markdown("### 🔊 Audio Response")
            audio_bytes = text_to_audio(ai_response)
            
            if audio_bytes:
                # Create columns for audio player and download button
                audio_col1, audio_col2 = st.columns([3, 1])
                
                with audio_col1:
                    st.audio(audio_bytes, format='audio/mp3', autoplay=True)
                
                with audio_col2:
                    st.download_button(
                        label="📥 Download",
                        data=audio_bytes,
                        file_name="ai_response.mp3",
                        mime="audio/mp3",
                        key="text_download"
                    )

with col2:
    st.subheader("📜 Chat History")
    
    # Add clear history button
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.chat_history = ChatMessageHistory()
        st.rerun()
    
    # Display chat history in a scrollable container
    if st.session_state.chat_history.messages:
        history_container = st.container(height=500)
        with history_container:
            for i, msg in enumerate(st.session_state.chat_history.messages):
                if msg.type == "human":
                    st.markdown(f"**👤 You:** {msg.content}")
                else:
                    st.markdown(f"**🤖 AI:** {msg.content}")
                
                if i < len(st.session_state.chat_history.messages) - 1:
                    st.markdown("---")
    else:
        st.info("No conversation history yet. Start chatting!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🚀 Powered by Ollama & LangChain | 🎙️ Speech Recognition by Google | 🔊 Text-to-Speech by gTTS</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar with Instructions
with st.sidebar:
    st.header("📖 Instructions")
    st.markdown("""
    ### How to use:
    1. **Voice Input:** Click on 'Record your voice' and speak clearly
    2. **Text Input:** Type your message in the text box
    3. **Audio Response:** The AI response will be played automatically
    4. **Download:** You can download the audio response as MP3
    
    ### Requirements:
    - Internet connection for speech recognition
    - Microphone access for voice input
    - Ollama running locally with Mistral model
    
    ### Tips:
    - Speak clearly and avoid background noise
    - Keep questions concise for better responses
    - Clear history if conversations get too long
    """)
    
    st.markdown("---")
    
    # Settings
    st.header("⚙️ Settings")
    
    # Language selection for TTS
    tts_lang = st.selectbox(
        "Text-to-Speech Language",
        options=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
        format_func=lambda x: {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese"
        }.get(x, x)
    )
    
    # Model selection
    model_name = st.selectbox(
        "Ollama Model",
        options=["mistral", "llama3", "codellama", "neural-chat"],
        help="Select the Ollama model to use"
    )
    
    if st.button("Apply Settings"):
        # Update model
        st.session_state.model_name = model_name
        st.success("Settings updated!")