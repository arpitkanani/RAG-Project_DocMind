import sys
import yaml
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from src.components.retriever import Retriever
from src.components.memory_manager import MemoryManager
from src.logger import logging
from src.exception import CustomException

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


def format_docs(docs: list) -> str:
    """Format retrieved chunks into single context string with citations."""
    if not docs:
        return "No relevant content found in uploaded documents."

    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'N/A')} | "
        f"Page: {doc.metadata.get('page', 'N/A')}]:\n"
        f"{doc.page_content}"
        for doc in docs
    )


QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are DocMind, an intelligent document assistant.
Answer questions strictly and only from the provided document context.
Never use your own training knowledge to answer.
If context does not contain the answer say:
"This information is not available in the uploaded document."
Always mention source and page naturally at end of answer.
Answer in the same language the user asks in.
Be concise and natural — never robotic or template-like.
Never use markdown bold headers like **text**.
Never leave answer incomplete."""),

    MessagesPlaceholder(variable_name="chat_history"),

    ("human", """Document context:
{context}

Question: {question}

Answer only from the context above:""")
])


def build_qa_chain(collection_name: str = None):
    """Build complete RAG Q&A chain using LCEL pipe."""
    try:
        logging.info("Building QA chain")

        llm = ChatOllama(
            model=config["llm"]["model"],
            temperature=config["llm"]["temperature"]
        )
        parser = StrOutputParser()

        retriever_obj = Retriever(collection_name=collection_name)
        retriever = retriever_obj.get_retriever_object()

        # RunnableParallel runs all 3 branches simultaneously
        # each branch receives same input dict
        # outputs merged into one dict → passed to prompt
        parallel_step = RunnableParallel({
            "context": (
                RunnableLambda(lambda x: x["question"])
                | retriever
                | RunnableLambda(format_docs)
            ),
            "question": RunnableLambda(
                lambda x: x["question"]
            ),
            "chat_history": RunnableLambda(
                lambda x: x["chat_history"]
            ),
        })

        chain = parallel_step | QA_PROMPT | llm | parser

        logging.info("QA chain built successfully")
        return chain

    except Exception as e:
        raise CustomException(e, sys)


def get_answer(question: str,
               collection_name: str = None) -> str:
    """Full Q&A flow — load memory, get answer, save memory."""
    try:
        logging.info(f"Processing question: {question[:50]}...")

        memory = MemoryManager()
        chat_history = memory.get_history()

        chain = build_qa_chain(collection_name)

        answer = chain.invoke({
            "question": question,
            "chat_history": chat_history
        })

        memory.save_message("human", question)
        memory.save_message("ai", answer)

        logging.info("Answer generated and saved to memory")
        return answer

    except Exception as e:
        raise CustomException(e, sys)