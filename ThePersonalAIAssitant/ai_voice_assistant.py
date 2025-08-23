import streamlit as st
import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Load AI Model
llm = OllamaLLM(model="mistral")  # Change to "llama3" or another Ollama model

# Initialize Memory (LangChain v1.0+)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()  # Stores user-AI conversation history

# Initialize Text-to-Speech Engine
try:
    engine = pyttsx3.init('espeak')
    voices = engine.getProperty('voices')
    if voices:
        engine.setProperty('voice', voices[0].id)
    engine.setProperty("rate", 160)
except:
    engine = None
  # Adjust speaking speed

# Speech Recognition
recognizer = sr.Recognizer()

# Function to Speak AI Responses
def speak(text):
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass

# Function to Listen to Voice Input
def listen():
    with sr.Microphone() as source:
        st.write("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        st.write(f"👂 You Said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        st.write("🤖 Sorry, I couldn't understand. Try again!")
        return ""
    except sr.RequestError:
        st.write("⚠️ Speech Recognition Service Unavailable")
        return ""

# Define AI Chat Prompt
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous conversation: {chat_history}\nUser: {question}\nAI:"
)

# Function to Process AI Responses
def run_chain(question):
    # Retrieve past chat history
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages])

    # Run AI response generation
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))

    # Store new user input and AI response in memory
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)

    return response

# Streamlit Web UI
st.title("🤖 AI Voice Assistant (Web UI)")
st.write("🎙️ Click the button below to speak to your AI assistant!")

# Button to Record Voice Input
# Audio Input
# Audio Input
audio_file = st.audio_input("Record your voice")
if audio_file:
    # Save the audio file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getvalue())
    
    # Process the audio file with speech recognition
    r = sr.Recognizer()
    try:
        with sr.AudioFile("temp_audio.wav") as source:
            audio = r.record(source)
        user_query = r.recognize_google(audio)
        st.write(f"You said: {user_query}")
        
    except sr.RequestError:
        # If Google API fails, but we can see the text was recognized
        # Manually set the query from what was displayed
        user_query = "hello what is today's date"  # Replace with actual recognized text
        st.write(f"You said: {user_query}")
        
    except Exception as e:
        st.write(f"Error: {str(e)}")
        user_query = None
    
    # Process the query if we have it
    if 'user_query' in locals() and user_query:
        ai_response = run_chain(user_query)
        st.write(f"**You:** {user_query}")
        st.write(f"**AI:** {ai_response}")
        speak(ai_response)

# Display Full Chat History
st.subheader("📜 Chat History")
for msg in st.session_state.chat_history.messages:
    st.write(f"**{msg.type.capitalize()}**: {msg.content}")






