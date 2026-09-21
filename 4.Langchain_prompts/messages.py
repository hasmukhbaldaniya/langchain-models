from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello! How are you?"),
    AIMessage(content="I'm doing well, thank you! How can I assist you today?"),
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)