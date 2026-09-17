from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_completion_tokens=100)

result = chat_model.invoke("Hello, how are you?")

print(result)