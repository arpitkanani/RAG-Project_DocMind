import sys
import yaml
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from src.components.vector_store import VectorStore
from src.utils.intent_detector import extract_quiz_count
from src.logger import logging
from src.exception import CustomException

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


DEFAULT_QUIZ_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are DocMind, a quiz generation assistant.
Generate questions ONLY from the provided document context.
Never generate questions about topics not in the document.
Each question must have a clear answer found in the document.
Generate in same language as the document content.
also make sure answer is not too short like one line answer or too long like 10 line answer if user not specifically tell about answer length in question then keep 3-4 line answer .
If user specifically tell about answer length in question like short answer or long answer then follow that instruction

     """),

    ("human", """Document context:
{context}

Generate a quiz with exactly:
- {easy_count} Easy questions (basic recall from document)
- {medium_count} Medium questions (understanding concepts)
- {hard_count} Hard questions (applying or analyzing content)

Format exactly like this:
- Answer of question as you want length if specifically not tell in question length like short ,long or etc.

EASY:
Q1. [question]
A1. [answer from document]

Q2. [question]
A2. [answer from document]

MEDIUM:
Q3. [question]
A3. [answer from document]

HARD:
Q. [question]
A. [answer from document]

Generate quiz now:""")
])

CUSTOM_QUIZ_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are DocMind, a quiz generation assistant.
Generate questions ONLY from the provided document context.
Never generate questions about topics not in the document.
Each question must have a clear answer found in the document.
Generate in same language as the document content.
also make sure answer is not too short like one line answer or too long like 10 line answer if user not specifically tell about answer length in question then keep 3-4 line answer .
If user specifically tell about answer length in question like short answer or long answer then follow that instruction

     """),

    ("human", """Document context:
{context}

Generate exactly {count} questions with answers.
Mix difficulty levels naturally across all questions.

Format exactly like this:
Q1. [question]
A1. [answer from document]

Q2. [question]
A2. [answer from document]

Continue for all {count} questions.

Generate quiz now:""")
])

class QuizChain:
    """Generates quiz questions from entire document content."""

    def __init__(self):
        """Initialize LLM, parser and both quiz chains."""
        try:
            logging.info("Initializing QuizChain")

            self.llm = ChatOllama(
                model=config["llm"]["model"],
                temperature=config["llm"]["temperature"]
            )
            self.parser = StrOutputParser()

            self.default_chain = (
                DEFAULT_QUIZ_PROMPT | self.llm | self.parser
            )
            self.custom_chain = (
                CUSTOM_QUIZ_PROMPT | self.llm | self.parser
            )

            logging.info("QuizChain initialized successfully")

        except Exception as e:
            raise CustomException(e, sys)#type:ignore
        
    def _get_full_context(self,
                          collection_name: str = None) -> str:#type:ignore
        try:
            vs = VectorStore(collection_name=collection_name)
            all_docs = vs.get_all_documents()

            if not all_docs:
                return ""

            # join all chunks with clear separator
            # LLM sees full document as continuous text
            context = "\n\n---\n\n".join(
                doc.page_content
                for doc in all_docs
                if doc.page_content.strip()
            )
            logging.info(
                f"Full context built: {len(all_docs)} chunks | "
                f"{len(context)} total chars"
            )
            return context

        except Exception as e:
            raise CustomException(e, sys)#type:ignore
        
    def generate(self,
                 query: str,
                 collection_name: str = None) -> str:#type:ignore
        """
        Main entry point for quiz generation.

        Detects if user specified count or wants default quiz.
        Routes to correct chain accordingly."""
        try:
            logging.info(f"Generating quiz for: {query[:50]}...")

            # get full document context
            context = self._get_full_context(collection_name)

            if not context:
                return "No document found. Please upload a document first."

            # extract_quiz_count returns 0 if no number in query
            # 0 means user wants default quiz
            count = extract_quiz_count(query)

            if count > 0:
                # user specified exact count
                logging.info(f"Custom quiz: {count} questions")
                quiz = self._generate_custom(context, count)
            else:
                # use default easy/medium/hard from config
                logging.info("Default quiz: easy/medium/hard")
                quiz = self._generate_default(context)

            logging.info("Quiz generated successfully")
            return quiz

        except Exception as e:
            raise CustomException(e, sys)#type:ignore
        
    def _generate_default(self, context: str) -> str:
        """
        Generate default quiz using easy/medium/hard counts from config.

        reads counts from config:
        easy   = 5 (default)
        medium = 3 (default)
        hard   = 2 (default)
        total  = 10 questions
        """
        try:
            easy_count   = config["quiz"]["default_easy"]
            medium_count = config["quiz"]["default_medium"]
            hard_count   = config["quiz"]["default_hard"]

            logging.info(
                f"Default quiz: {easy_count} easy | "
                f"{medium_count} medium | "
                f"{hard_count} hard"
            )

            quiz = self.default_chain.invoke({
                "context":      context,
                "easy_count":   easy_count,
                "medium_count": medium_count,
                "hard_count":   hard_count,
            })

            return quiz

        except Exception as e:
            raise CustomException(e, sys)

    def _generate_custom(self,
                         context: str,
                         count: int) -> str:
        """
        Generate quiz with exact number of questions user requested.

        count comes from extract_quiz_count()
        already capped at max_questions from config
        so no need to validate here
        """
        try:
            logging.info(f"Custom quiz: generating {count} questions")

            quiz = self.custom_chain.invoke({
                "context": context,
                "count":   count,
            })

            return quiz

        except Exception as e:
            raise CustomException(e, sys)#type:ignore



def get_quiz(query: str, 
             collection_name: str = None) -> str: 
        """Static method to generate quiz without needing to instantiate."""
        try:
            logging.info(f"Quiz request: {query[:50]}...")
            quiz_chain = QuizChain()
            result = quiz_chain.generate(query, collection_name)
            logging.info("Quiz returned successfully")
            return result

        except Exception as e:
            raise CustomException(e, sys)#type:ignore