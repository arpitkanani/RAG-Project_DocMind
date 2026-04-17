import sys
import re
import yaml
from src.logger import logging
from src.exception import CustomException

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

INTENT_QA      = "qa"
INTENT_SUMMARY = "summary"
INTENT_QUIZ    = "quiz"

SUMMARY_KEYWORDS = config["intent"]["summary_keywords"]
QUIZ_KEYWORDS    = config["intent"]["quiz_keywords"]
DETAIL_KEYWORDS  = config["intent"]["detail_keywords"]

def detect_intent(query: str) -> str:
    """
    Detects user intent based on keywords in the query.
    """
    try:
        query_lower = query.lower().strip()
        logging.info(f"Detecting intent for: {query_lower[:50]}")

        if _is_quiz_intent(query_lower):
            logging.info("Intent detected: QUIZ")
            return INTENT_QUIZ
        
        if _is_summary_intent(query_lower):
            logging.info("Intent detected: SUMMARY")
            return INTENT_SUMMARY
        
        logging.info("Intent detected: QA")
        return INTENT_QA
    
    except Exception as e:
        raise CustomException(e, sys) #type:ignore
    

def _is_quiz_intent(query_lower: str) -> bool:
    try:
        keyword_match=any(
            keyword in query_lower for keyword in QUIZ_KEYWORDS
        )

        if keyword_match:
            return True
        
        pattern=re.search(
            r"(generate|create|make|give)\s+\d+\s+question",
            query_lower
        )

        return pattern is not None
    
    except Exception as e:
        raise CustomException(e, sys) #type:ignore
    
def _is_summary_intent(query_lower: str) -> bool:
    try:
        return any(
            keyword in query_lower
            for keyword in SUMMARY_KEYWORDS
        )
    except Exception as e:
        raise CustomException(e, sys)#type:ignore

def detect_summary_type(query:str)->str:

    try:
        query_lower=query.lower().strip()

        if any(keyword in query_lower for keyword in DETAIL_KEYWORDS):
            return "structured"
        
        if "brief" in query_lower or "short" in query_lower:
            return "brief"
        
        return "bullets"
    
    except Exception as e:
        raise CustomException(e, sys)#type:ignore
    

def extract_quiz_count(query:str)->int:
    try:
        query_lower=query.lower().strip()
        match=re.search(r"\d+", query_lower)

        if match:
            count=int(match.group())
            max_q=config["quiz"]["max_questions"]
            if count > max_q:
                logging.warning(
                    f"Requested {count} questions exceeds "
                    f"max {max_q}. Using max."
                )
                return max_q
        
            logging.info(f"Quiz count extracted: {count}")
        
            return count
        logging.info("No quiz count in query, using default")
        return 0
    except Exception as e:
        CustomException(e, sys)#type:ignore


        

    