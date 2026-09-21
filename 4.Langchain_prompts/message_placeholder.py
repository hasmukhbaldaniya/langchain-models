from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("user", "{text}"),     
    MessagesPlaceholder(variable_name="history")
])

chat_history = []

with open("chat_history.txt") as file:
    for line in file:
        if line.startswith("User:"):
            user_input = line[len("User:"):].strip()
            chat_history.append(("user", user_input))
        elif line.startswith("Assistant:"):
            assistant_response = line[len("Assistant:"):].strip()
            chat_history.append(("assistant", assistant_response))

history = chat_template.invoke({'history': chat_history, 'input_language': 'English', 'output_language': 'Spanish', 'text': 'Hello, how are you?'})

print(history)