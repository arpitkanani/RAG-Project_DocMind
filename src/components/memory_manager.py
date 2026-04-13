import json
import sys
import os
import yaml
from datetime import datetime,timedelta
from langchain_core.messages import HumanMessage,BaseMessage,AIMessage
from src.logger import logging
from src.exception import CustomException
from typing import List


with open("D:\Langchain Project\config\config.yaml") as f:
    config = yaml.safe_load(f)

class MemoryManager:
    """Manages persistent 7 day chat history"""
    def __int__(self):
        try:
            logging.info("Initializing MemoryManager")

            self.persist_dir=config['memory']['persist_directory']
            self.window_days=config['memory']['"window_days']

            os.makedirs(self.persist_dir, exist_ok=True)

            self.memory_file = os.path.join(
                self.persist_dir,
                "chat_history.json"
            )
            logging.info(
                f"MemoryManager ready | "
                f"file: {self.memory_file} | "
                f"window: {self.window_days} days"
            )

        except Exception as e:
            raise CustomException(e, sys)#type:ignore
        
    def save_message(self,role:str,content:str):

        """Save single message to JSON file"""
        try:
            history=self._load_raw()

            history.append({
                "role":role,
                "content":content,
                "timestamp":datetime.now().isoformat()
            })

            with open(self.memory_file,"w",encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

            logging.info(f"Message saved | role: {role}")
        except Exception as e:
            raise CustomException(e,sys)#type:ignore
        

    def get_history(self) -> List[BaseMessage]:
        """
        Load chat history from last 7 days
        Returns list of LangChain message objects
        """
        try:
            raw_history = self._load_raw()

            cutoff =datetime.now() - timedelta(days=self.window_days)

            messages=[]

            for msg in raw_history:

                msg_time=datetime.fromisoformat(msg['timestamp'])

                if msg['role']== "human":
                    messages.append(
                        HumanMessage(content=msg['content'])
                    )

                elif msg['role']=="ai":
                    messages.append(
                        AIMessage(content=msg['content'])
                    )

                logging.info(
                     f"History loaded: {len(messages)} messages "
                    f"from last {self.window_days} days"
                )

                return messages
            
        except Exception as e:
            raise CustomException(e,sys)#type:ignore
        
    
    def get_history_as_text(self)-> str:
        """
        Return chat history as plain formatted string
        whwn needed.
        """
        try:

            messages=self.get_history()

            if not messages:
                return "No conversation history yet."
            
            lines=[]

            for msg in messages:
                if isinstance(msg,HumanMessage):
                    lines.append(f"Human:{msg.content}")
                
                elif isinstance(msg,AIMessage):
                    lines.append(f"AI:{msg.content}")
            

            return "\n".join(lines)
        
        except Exception as e:
            raise CustomException(e, sys)#type:ignore
        
    
    def clear(self):
        """
        Delete entire chat history
        """
        try:
            if os.path.exists(self.memory_file):
                os.remove(self.memory_file)
            logging.info("Chat history cleared")

        except Exception as e:
            raise CustomException(e, sys)#type:ignore
        
    def get_message_count(self) -> int:
        """
        Return total messages in current window

        WHY THIS FUNCTION?
        useful for frontend to show
        "You have X messages in history"
        also useful for deciding when to
        summarize old history to save tokens
        """
        try:
            return len(self.get_history())
        except Exception as e:
            raise CustomException(e, sys)#type:ignore
        
    def _load_raw(self) -> list:
        """
        Private helper → load raw JSON from disk
        """
        try:
            if not os.path.exists(self.memory_file):
                return []

            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f)

        except Exception as e:
            raise CustomException(e, sys)#type:ignore
        
        