from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

model = ChatOpenAI()

# 1st prompt -> detailed report
prompt1 = PromptTemplate(
    template="Generate 5 detailed report on {topic}",
    input_variables=["topic"]
)

# 2nd prompt -> summary
prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "Unemployment in india"})

print(result)

chain.get_graph().print_ascii()