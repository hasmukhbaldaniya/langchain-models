from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Literal, Optional
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

# schema
class Review(BaseModel):
    key_themes: list[str] = Field(
        description="Write down all the key themes discussed in the review in a list"
    )

    summary: str = Field(
        description="A brief summary of the review"
    )

    sentiment: Literal["pos", "neg"] = Field(
        description="Return sentiment of the review either negative, positive or neutral"
    )

    pros: Optional[list[str]] = Field(
        default=None,
        description="Write down all the pros inside a list"
    )

    cons: Optional[list[str]] = Field(
        default=None,
        description="Write down all the cons inside a list"
    )

    name: Optional[str] = Field(
        default=None,
        description="Write the name of the reviewer"
    )

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    """I recently upgraded to the Samsung Galaxy S24 Ultra, and
I am very impressed with the camera quality and display. The phone
feels premium and performs very well. However, it is quite expensive
and the phone is also a little heavy. The battery life is good, but
I expected slightly better battery performance for the price."""
)

print(result)