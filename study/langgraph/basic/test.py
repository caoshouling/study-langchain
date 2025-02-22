from typing import Optional, TypedDict, Annotated

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1/",
    # model="paultimothymooney/qwen2.5-7b-instruct", qwen2.5-14b-instruct
    model = "triangle104/qwen2.5-7b-instruct",
    api_key="323"
)
# Pydantic
class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


# TypedDict
class Joke2(TypedDict):
    """Joke to tell user."""

    setup: Annotated[str, ..., "The setup of the joke"]

    # Alternatively, we could have specified setup as:

    # setup: str                    # no default, no description
    # setup: Annotated[str, ...]    # no default, no description
    # setup: Annotated[str, "foo"]  # default, no description

    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]


structured_llm = model.with_structured_output(Joke2)


print(structured_llm.invoke("Tell me a joke about cats"))