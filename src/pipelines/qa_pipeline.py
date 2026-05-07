import yaml
import sys
import os
from src.logger import logging
from src.exception import CustomException
from src.chains.qa_chain import get_answer
from src.chains.summary_chain import get_summary
from src.chains.quiz_chain import get_quiz
from src.utils.intent_detector import (
    detect_intent,
    INTENT_SUMMARY,
    INTENT_QUIZ,
    INTENT_QA

)

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


class QAPipeline:
    """ routes user query based on intent to appropriate chain and manages memory """


    def __init__(self,colllection_name:str=None):
       

        logging.info("Initializing QAPipeline")
        self.collection_name = self.collection_name

    def run(self,query:str)-> dict:
        """
        main entry point for every query.
        """

        try:
            logging.info(f"Summary request: {query[:50]}...")
            
            intent=detect_intent(query)
            if intent==INTENT_SUMMARY:
                
                answer = get_summary(
                    query,
                    collection_name=self.collection_name
                )
            elif intent==INTENT_QUIZ:
                
                answer = get_quiz(
                    query,
                    collection_name=self.collection_name
                )
            else:
                
                answer = get_answer(
                    query,
                    collection_name=self.collection_name
                
                )
            
            result ={
                "answer":answer,
                "intent":intent,
                "collection":self.collection_name or "all",
                "query":query

            }

            logging.info(f"Query Processed |Intent: {intent}")
            return result
        except Exception as e:
            raise CustomException(e, sys)#type:ignore
        