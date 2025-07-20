import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Load AI Model
llm = OllamaLLM(model="mistral")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous conversation: {chat_history}\nUser: {question}\nAI:"

)

# Function to run AI chat with memory
def run_chain(question):
    # Retrieve past chat history
    chat_history_text ="n".join(
        [f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages]
    )
    
    # Generate response or AI response generation
    response = llm.invoke(prompt.format(
        chat_history=chat_history_text,
        question=question
    ))
    # Store new user input and AI response in memory
    st.session_state.chat_history.add_user_message(question)
    
    # Add AI response to chat history
    st.session_state.chat_history.add_ai_message(response)
    
    return response


# Streamlit UI
st.title("AI Agent with Memory")
st.write("This is a simple AI agent that remembers past conversations.\n Ask me anything!")
user_input = st.text_input("Your question:")
if user_input:
    response = run_chain(user_input)
    st.write("You asked:", user_input)
    st.write("AI Response:", response)

# show full chat history
st.subheader("Chat History")
for msg in st.session_state.chat_history.messages:
    st.write(f"**{msg.type.capitalize()}**: {msg.content}")