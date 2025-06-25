from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="llama3")

while True:
    user_input = input("Your question (or type 'exit' to stop): ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat.")
        break
    response = llm.invoke(user_input)
    print(f"AI: {response}")