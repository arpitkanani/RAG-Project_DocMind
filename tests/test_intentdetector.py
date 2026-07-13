import sys
import os
sys.path.append(os.getcwd())

from src.utils.intent_detector import (
    detect_intent,
    detect_summary_type,
    extract_quiz_count,
    INTENT_QA,
    INTENT_SUMMARY,
    INTENT_QUIZ
)

print("TESTING — INTENT DETECTOR")


# test cases — query and expected intent
test_cases = [
    ("what is machine learning?",           INTENT_QA),
    ("summarize this document",             INTENT_SUMMARY),
    ("give me a brief summary",             INTENT_SUMMARY),
    ("generate 10 questions",               INTENT_QUIZ),
    ("make a quiz from this",               INTENT_QUIZ),
    ("create 5 questions about neural nets",INTENT_QUIZ),
    ("what did he say about RAG?",          INTENT_QA),
    ("detailed summary please",             INTENT_SUMMARY),
    ("test me on this content",             INTENT_QUIZ),
]

print("\n--- Intent Detection ---")
all_passed = True
for query, expected in test_cases:
    result = detect_intent(query)
    status = "true" if result == expected else "false"
    if result != expected:
        all_passed = False
    print(f"{status} '{query}'")
    print(f"   Expected: {expected} | Got: {result}\n")

print("--- Summary Type Detection ---")
summary_tests = [
    ("summarize this",          "bullets"),
    ("brief summary",           "brief"),
    ("detailed summary",        "structured"),
    ("comprehensive overview",  "structured"),
    ("short summary",           "brief"),
    ("what is this document about?","bullets"),
]
for query, expected in summary_tests:
    result = detect_summary_type(query)
    status = "true" if result == expected else "false"
    print(f"{status} '{query}' → {result}")

print("\n--- Quiz Count Extraction ---")
count_tests = [
    ("generate 10 questions",  10),
    ("make 5 questions",        5),
    ("create 20 questions",    20),
    ("generate quiz",           0),
    ("make questions",          0),
]
for query, expected in count_tests:
    result = extract_quiz_count(query)
    status = "true" if result == expected else "false"
    print(f"{status} '{query}' → {result} questions")

print("\n" + "-" * 50)
if all_passed:
    print("ALL INTENT TESTS PASSED!")
else:
    print("SOME TESTS FAILED — check above")
