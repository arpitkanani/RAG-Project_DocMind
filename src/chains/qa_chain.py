import sys
import yaml
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda

from src.components.memory_manager import MemoryManager
from src.components.vector_store import VectorStore
from src.logger import logging
from src.exception import CustomException

with open("config/config.yaml") as f:
    config=yaml.safe_load(f)

def format_docs(docs:list) -> str:

    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'N/A')} | "
        f"Page: {doc.metadata.get('page', 'N/A')}]:\n"
        f"{doc.page_content}"
        for doc in docs
    )

def build_qa_chain(collection_name:str=None):
    try:
        logging.info("Building QA Chain")

        llm=ChatOllama(model=config["ollama"]["model"],
                        temperature=config["ollama"]["temperature"])
        
        parser=StrOutputParser()

        vs=VectorStore()

        retriever=vs.get_retriever(collection_name=collection_name)

        prompt=ChatPromptTemplate([
            ("system", """You are DocMind, an intelligent document assistant.
                Your job is to answer questions based ONLY on the provided context.
                Always mention the source and page number when answering.
                If the answer is not found in context, say exactly:
                'I could not find this information in the document.'
                Be concise, accurate and helpful."""),

                MessagesPlaceholder(variable_name="chat_history"),

                ("human","""
                    context:{context}
                    question:{question}
                    Answer based on strictly on the context above:
                """)
        ])

        chain =(
            {
                "context":(
                    RunnableLambda(lambda x:x['question'])|retriever |RunnablePassthrough(format_docs) # type: ignore
                ),
                "question":RunnableLambda(lambda x:x["question"]), # type: ignore
                "chat_history":RunnableLambda(lambda x:x["chat_history"]),
            }
            | prompt | llm | parser
        )

        logging.info("QA chain built successfully")
        return chain
    except Exception as e:
        raise CustomException(e,sys)#type:ignore
    

def get_answer(question:str,collection_name:str ->None)->str:

    try:
        logging.info(f"Processing question: {question[:50]}...")

        memory=MemoryManager()

        chat_history=memory.get_history()

        chain=build_qa_chain(collection_name=collection_name)

        answer=chain.invoke({
            "question":question,
            "chat_history":chat_history
        })

        memory.save_message("human",question)
        memory.save_message('ai',answer)

        logging.info("Answer generated and saved to memory")
        return answer

    except Exception as e:
        raise CustomException(e, sys)#type:ignore
    
    
