from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv  

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.8-flash",)

result = chat_model.invoke("Hello, how are you?")

print(result.content)