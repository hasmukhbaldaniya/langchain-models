from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful {domain} expert."),
    ('user', "Explain the simple terms of {topic}"),
])

prompt = chat_template.invoke({"domain": "cricket", "topic": "spin bowling"})

print(prompt.messages)  # This will print the list of messages in the prompt