from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

model = ChatOpenAI()


# schema
class Review(TypedDict):
    key_themes: Annotated[
        list[str],
        "Write down all the key themes discussed in the review in a list",
    ]
    summary: Annotated[
        str,
        "A brief summary of the review",
    ]
    sentiment: Annotated[
        str,
        "Return sentiment of the review either negative, positive or neutral",
    ]
    pros: Annotated[
        Optional[list[str]],
        "Write down all the pros inside a list",
    ]
    cons: Annotated[
        Optional[list[str]],
        "Write down all the cons inside a list",
    ]


structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    """I recently upgraded to the Samsung Galaxy S24 Ultra, and
I am very impressed with the camera quality and display. The phone
feels premium and performs very well. However, it is quite expensive
and the phone is also a little heavy. The battery life is good, but
I expected slightly better battery performance for the price."""
)

print(result)
print(result["key_themes"])
print(result["summary"])
print(result["sentiment"])
print(result["pros"])
print(result["cons"])