import sys
import os
sys.path.append(os.getcwd())

from src.components.memory_manager import MemoryManager

memory = MemoryManager()

memory.clear()  # fresh start

memory.save_message("human", "What is machine learning?")
memory.save_message("ai", "ML is a subset of AI that learns from data.")
memory.save_message("human", "Give me a real world example.")
memory.save_message("ai", "Spam detection in Gmail is a great example.")



history = memory.get_history()
print(f"Messages loaded: {len(history)}")
for i, msg in enumerate(history):
    print(f"\nMessage {i+1}:")
    print(f"Type    : {type(msg).__name__}")
    print(f"Content : {msg.content}")


print(memory.get_history_as_text())

print(f"Total messages: {memory.get_message_count()}")


memory.clear()
print(f"After clear: {memory.get_message_count()} messages")
print(f"History: {memory.get_history_as_text()}")
