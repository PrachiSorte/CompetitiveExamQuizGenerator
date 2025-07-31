#This python file define helper function (reusablility)

# import required libraries
import os
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  # for LLM model Components

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser  # validate questions, users answers and Response from LLM
from pydantic import BaseModel, Field, field_validator

# Load environment variables from .env file
load_dotenv()

# Define data model for Multiple Choice Questions using Pydantic
class MCQQuestion(BaseModel):
    # Define the structure of an MCQ with field description
    question: str=Field(description="The question text")
    options:List[str] = Field(description="List of 4 possible answers")
    correct_answer:str =Field(description="The correct answer from the options")

# Custom validator to clean question text
# Handle cases where question might be a dictionary or other format

@field_validator('question',mode='before')
def clean_question(cls,v): #{"description":42}-> "42"
    if isinstance(v,dict):
        return v.get('description',str(v))
    return str(v)

# Define data model for Fill in the Blank Questions using Pydantic
class FillBlankQuestion(BaseModel):
    # Define the structure of a Fill In the blank questions with field description
    question: str=Field(description="The question text with '_____________' for the blank")
    answer:str =Field(description="The correct word or phrase for the blank")

# Custom validator to clean question text
# similar to MCQ validator, ensures consistent question formatcmc

@field_validator('question',mode='before')
def clean_question(cls,v): #{"description":42}-> "42"
    if isinstance(v,dict):
        return v.get('description',str(v))
    return str(v)



# Generate MCQ Questions
class QuestionGenerator:
    def __init__(self):
        
        """
        Initialize question generator with Grog API
        """

        #self.llm= ChatGroq(
        #    api_key=os.getenv('GROQ_API_KEY'),
        #    model="llama-3.1-8b-instant",
        #    temperature=0.9
        #)

        self.llm=ChatGoogleGenerativeAI(
           api_key=os.getenv('GOOGLE_API_KEY'),    
           model="gemini-1.5-flash",
           temperature=0.9
        )

        

            


    def generate_mcq(self, topic:str, difficulty: str="medium") -> MCQQuestion:

        # setup pydantic parser for type checking and validation
        mcq_parser=PydanticOutputParser(pydantic_object=MCQQuestion)

        #define prompt template
        prompt=PromptTemplate(
            template=
                """
                Generate a {difficulty} multiple-choice question about {topic}.
                Do not repeat the same questions.
                Return output with these exact fields:
                'question' : A clear, specific question
                'options': An array of exactly 4 possible answers
                'correct_answer': one of the options that is the correct answer
                Example format:
                
                 "question": "What is the capital of France?",
                 "options": ["London", "Berlin", "Paris", "Madrid"],
                 "correct_answer": "Paris"
                 
                Your Response:
                
                """
            ,
            input_variables=["topic","difficulty"]
            
        )

        #implement retry logic with maximum attempts
        max_attempts=3

        for attempt in range(max_attempts):
            try:
                # Generate response using LLM
                response=self.llm.invoke(prompt.format(topic=topic,difficulty=difficulty))
                parsed_response=mcq_parser.parse(response.content)

                #validate the generated questions meet requirements
                if not parsed_response.question or len(parsed_response.options)!=4 or not parsed_response.correct_answer:
                    raise ValueError("Invalid Question Format")
                if parsed_response.correct_answer not in parsed_response.options:
                    raise ValueError("Correct answer is not in options")
                
                return parsed_response
            except Exception as e:

                #On final attempt, raise error, otherwise continue trying
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate valid MCQ after {max_attempts} attempts: {str(e)}")
                continue





    # Generate Fill-in-the-blank Questions
    def generate_fill_blank(self, topic:str, difficulty: str="medium") -> FillBlankQuestion:

        # setup pydantic parser for type checking and validation
        fillblank_parser=PydanticOutputParser(pydantic_object=FillBlankQuestion)

        #define prompt template
        prompt=PromptTemplate(
           template=
                """
                Generate a {difficulty} fill-in-the-blank question about {topic}. 
                Generate each question uniquely. Do not repeat the same questions.
                Return generated question with these exact fields:
                'question' :A sentence with '______' marking where the blank should be
                'answer': The correct word or phrase that belongs in the blank
                Example format:
                "question":"The capital of France is ______",
                "answer":"Paris"
                 
                Your Response:
                
                """
            ,
            input_variables=["topic","difficulty"]
            
        )

        #implement retry logic with maximum attempts
        max_attempts=3

        for attempt in range(max_attempts):
            try:
                # Generate response using LLM
                response=self.llm.invoke(prompt.format(topic=topic,difficulty=difficulty))
                parsed_response=fillblank_parser.parse(response.content)

                #validate the generated questions meet requirements
                if not parsed_response.question or not parsed_response.answer:
                    raise ValueError("Invalid Question Format")
                if  "______" not in parsed_response.question:
                    parsed_response.question=parsed_response.question.replace("___","_________")
                    if "_____" not in parsed_response.question:   
                        raise ValueError("Question missing blank marker '______'")
                
                return parsed_response
            except Exception as e:

                #On final attempt, raise error, otherwise continue trying
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate valid fill-in-the-blank question after {max_attempts} attempts: {str(e)}")
                continue











