import json
import asyncio
import os



from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


load_dotenv()

# print(os.getenv("GOOGLE_API_KEY"))

class Paper(BaseModel):
    title: str 
    authors: list[str] 
    year: int
    abstract: str = Field(description="The abstract of the paper")
    publication_date: str = Field( description="The publication date of the paper")
    journal: str = Field(description="The journal in which the paper was published")

class Formula(BaseModel):
    name: str
    latex: str = Field(description="The LaTeX representation of the formula")
    description: str = Field(description="A brief description of the formula")
    formula: str = Field(description="The mathematical representation of the formula")

class Trend(BaseModel):
    topic: str
    description: str = Field(description="A brief description of the trend")
    examples: list[str] = Field(description="Examples of the trend in action")

class Report(BaseModel):
    topic: str
    papers: list[Paper]
    formulas: list[Formula]
    trends: list[Trend]


async def main():

    topic = input("Enter a topic for the research report: ").strip()
    research_questions = input("What are the key research questions: ").strip()
    timeframe = input("What is the timeframe for the research report: ").strip()

    task = f"""Topic:{topic}
Research Questions:{research_questions}
Timeframe:{timeframe}

Gather 5-10 highly relevant research papers, formulas, and trends related to the topic.
Then identify the most important papers, formulas, and trends, and provide a summary of each.
Populate the report schema fully"""
    
    model = init_chat_model("gemini-3.1-flash-lite", model_provider="google-genai", ).with_structured_output(Report)

    result =await model.ainvoke([
        {'role': 'system', 'content': 'You are a research assistant that specializes in gathering and summarizing information on various topics.'},
        {'role': 'user', 'content': task}
    ])

    print(json.dumps(result.model_dump(), indent=4))

asyncio.run(main())