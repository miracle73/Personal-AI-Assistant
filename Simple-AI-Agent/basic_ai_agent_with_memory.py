from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Load AI Model
llm = OllamaLLM(model="mistral")
chat_history = ChatMessageHistory()

# More explicit prompt template
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""You are an AI assistant with memory of our conversation. 

CONVERSATION HISTORY:
{chat_history}

CURRENT QUESTION: {question}

INSTRUCTIONS: 
- Remember and reference previous parts of our conversation when relevant
- Be consistent with information you've shared before
- If this is the first message, introduce yourself briefly

RESPONSE:"""
)

def run_chain(question):
    # Format chat history more explicitly
    if chat_history.messages:
        history_parts = []
        for i, msg in enumerate(chat_history.messages):
            if msg.type == "human":
                history_parts.append(f"User: {msg.content}")
            else:
                history_parts.append(f"AI: {msg.content}")
        chat_history_text = "\n".join(history_parts)
    else:
        chat_history_text = "No previous conversation."
    
    # Create the full prompt
    full_prompt = prompt.format(chat_history=chat_history_text, question=question)
    
    # Get response
    response = llm.invoke(full_prompt)
    
    # Store in history
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)
    
    return response

# Debug function to see what's being sent to the model
def debug_prompt(question):
    if chat_history.messages:
        history_parts = []
        for msg in chat_history.messages:
            if msg.type == "human":
                history_parts.append(f"User: {msg.content}")
            else:
                history_parts.append(f"AI: {msg.content}")
        chat_history_text = "\n".join(history_parts)
    else:
        chat_history_text = "No previous conversation."
    
    full_prompt = prompt.format(chat_history=chat_history_text, question=question)
    print("=== DEBUG: Prompt being sent to model ===")
    print(full_prompt)
    print("=== END DEBUG ===\n")

print("\n🤖 AI Chatbot with Memory")
print("Type 'exit' or 'quit' to stop.")
print("Type 'debug' to see the prompt being sent to the model.\n")

while True:
    user_input = input("Your question: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye! 👋")
        break
    elif user_input.lower() == "debug":
        debug_prompt("What's my name?")  # Example debug
        continue
    
    response = run_chain(user_input)
    print(f"AI: {response}\n")