from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(model="claude-2")

result = llm.invoke("Hello, how are you?")

print(result.content)