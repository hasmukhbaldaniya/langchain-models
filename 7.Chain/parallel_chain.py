from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatOpenAI()
model2 = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# 1st prompt -> detailed report
prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"]
)

# 2nd prompt -> summary
prompt2 = PromptTemplate(
    template="Generate a 5 short question answers from the following text \n {text}",
    input_variables=["text"]
)

# 3rd prompt -> question answers
prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
}) | prompt3 | model2 | parser

text = "Unemployment is a major issue in India, with millions of people struggling to find work. The government has implemented various schemes to address this problem, but challenges remain. Factors contributing to unemployment include a growing population, lack of skill development, and economic slowdowns. The youth are particularly affected, leading to social and economic consequences. Addressing unemployment requires a multi-faceted approach, including education, skill training, and job creation initiatives."

result = chain.invoke({"text": text})

print(result)

chain.get_graph().print_ascii()