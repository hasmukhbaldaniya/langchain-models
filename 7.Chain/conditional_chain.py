from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()

class FeedbackSentiment(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(..., description="The sentiment of the feedback, either 'positive' or 'negative'.")

parser = PydanticOutputParser(pydantic_object=FeedbackSentiment)

# 1st prompt
prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
# 2nd prompt
prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback: \n {feedback}",
    input_variables=["feedback"],
)
# 3rd prompt
prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback: \n {feedback}",
    input_variables=["feedback"],
)

str_parser = StrOutputParser()

classify_chain = prompt1 | model | parser

# RunnableBranch takes (condition, runnable) tuples; the last argument is the default
branch_chain = RunnableBranch(
    (lambda x: x["sentiment"].sentiment == "positive", prompt2 | model | str_parser),
    (lambda x: x["sentiment"].sentiment == "negative", prompt3 | model | str_parser),
    lambda x: "Could not determine the sentiment of the feedback.",
)

# Keep the original feedback alongside the classification so the branches can use it
chain = RunnablePassthrough.assign(sentiment=classify_chain) | branch_chain

print(chain.invoke({"feedback": "I dont like this product!"}))

chain.get_graph().print_ascii()