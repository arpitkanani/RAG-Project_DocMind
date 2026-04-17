import sys
import yaml
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel

from src.components.memory_manager import MemoryManager
from src.components.vector_store import VectorStore
from src.logger import logging
from src.exception import CustomException

with open("config/config.yaml") as f:
    config=yaml.safe_load(f)

def format_docs(docs:list) -> str:

    if not docs:
        return "No specific context found in document."
    
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'N/A')} | "
        f"Page: {doc.metadata.get('page', 'N/A')}]:\n"
        f"{doc.page_content}"
        for doc in docs
    )

def build_qa_chain(collection_name:str=None):
    try:
        logging.info("Building QA Chain")

        llm=ChatOllama(model=config["llm"]["model"],
                        temperature=config["llm"]["temperature"])
        
        parser=StrOutputParser()

        vs=VectorStore()

        retriever=vs.get_retriever(collection_name=collection_name)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are DocMind, a smart and helpful document assistant.
            You answer questions in a natural, conversational way — like a knowledgeable human, not a robot.

            How to answer:
            - Start directly with the answer, no labels or headers like "Direct Answer" or "Supporting Details"
            - Naturally weave in what the document says without explicitly labeling it
            - If the document covers it partially, smoothly continue with your knowledge to complete the picture
            - Only mention source like "(Source: filename, Page X)" also you can mention famouse reasearch paper or document about asked like link .once at the end if you used document content
            - If document has nothing relevant, just answer from knowledge naturally without making a big deal of it
            - Never use markdown bold (**text**) or bullet templates that feel robotic unless explicitaly told.
            - Never contradict yourself in the same answer
            - Keep it concise, complete and human sounding
            - Never say phrases like "Based on my general knowledge" or "I could not find" — just answer naturally
             1. if user ask Docmind user ask summarize the given data data may be pdf, text file , docx file . so you have to summarize it .
             2. if user ask generate quesiton based on given data then genrate as ask for it in query.
             and if answer is more big based on question you can give more concise it by saying 50% answer of it by your knowledge and relevant to context.
             if needed than make paragraph of answer to break large answer in to small by paragraphing with structured way.
             """),

                MessagesPlaceholder(variable_name="chat_history"),

                ("human", """Document context:
                    {context}

                    Question: {question}

                    Answer naturally and completely:""")
            ])

        parallel_chain=RunnableParallel({
            "context":(
                    RunnableLambda(lambda x:x['question'])|retriever | RunnablePassthrough(format_docs) # type: ignore
                ),
                "question":RunnableLambda(lambda x:x["question"]), # type: ignore
                "chat_history":RunnableLambda(lambda x:x["chat_history"]) # type: ignore
        })
        chain =   parallel_chain | prompt | llm | parser
        

        logging.info("QA chain built successfully")
        return chain
    except Exception as e:
        raise CustomException(e,sys)#type:ignore
    

def get_answer(question:str,collection_name:str =None)->str:

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
    
    
