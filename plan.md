# Human-in-the-Loop (HITL) Clarification & UI Implementation Guide

## 1. LLM System Prompt: Query Clarification & Structuring
*Inject this into your Query Analyzer or Routing Node so the model knows when and how to ask for clarification.*

**System Prompt:**
You are an intelligent query analyzer and routing agent for a document-retrieval (RAG) and conversational (Chitchat) system. Your primary goal is to evaluate the user's input for completeness and clarity before processing.

If a query is vague, ambiguous, misspelled, or insufficient to yield a highly accurate search or response (e.g., "cage", "APPL instead of AAPL", "metrics"), you MUST initiate a Human-in-the-Loop (HITL) feedback request instead of guessing.

When requesting clarification, you must output your response in the following strict JSON format so the frontend can render clickable options. Do not include markdown formatting around the JSON.

**Output Schema for Clarification:**
{
  "status": "requires_clarification",
  "message": "I need a bit more context to give you the best answer. Did you mean one of these?",
  "options": [
    "Option 1 (e.g., Nicholas Cage movies)",
    "Option 2 (e.g., Animal cage specifications)",
    "Option 3 (e.g., Cage (company) stock price)"
  ]
}

**Output Schema for Sufficient Queries:**
{
  "status": "sufficient",
  "intent": "rag_search | chitchat",
  "processed_query": "[The clean query to pass to the next node]"
}


## 2. Backend Graph Routing Logic (LangGraph / Flow)
*How to handle the model's output in your graph architecture.*

1. **Query Analyzer Node**: Passes the user input to the LLM with the prompt above.
2. **Conditional Edge**:
   - If `status == "requires_clarification"` -> Yield the response to the user and pause the graph (awaiting user feedback).
   - If `status == "sufficient" && intent == "rag_search"` -> Route to RAG/Vector Database node.
   - If `status == "sufficient" && intent == "chitchat"` -> Route to standard Chitchat LLM node.
3. **Feedback Resumption Node**: When the user submits their choice (either the clicked option or the manual text), inject this directly into the pipeline as the new, clarified `processed_query` and bypass the Query Analyzer.


## 3. Frontend Implementation (React / TailwindCSS)
*This code implements the Claude-style UI where options float above the input field. It enforces the mutual exclusivity: clicking an option disables manual typing, and typing disables the options.*

```jsx
import React, { useState } from 'react';

const ChatInputWithFeedback = ({ clarificationData, onSubmit }) => {
  const [manualInput, setManualInput] = useState('');
  const [selectedOption, setSelectedOption] = useState(null);

  // Handle typing in the text field
  const handleTextChange = (e) => {
    setManualInput(e.target.value);
    // If user starts typing, clear any selected option
    if (e.target.value.length > 0) {
      setSelectedOption(null);
    }
  };

  // Handle clicking a suggestion pill
  const handleOptionSelect = (option) => {
    // Toggle selection
    if (selectedOption === option) {
      setSelectedOption(null);
    } else {
      setSelectedOption(option);
      setManualInput(''); // Clear text field when an option is selected
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const finalQuery = selectedOption || manualInput;
    if (!finalQuery.trim()) return;
    
    onSubmit(finalQuery);
    
    // Reset state after submission
    setManualInput('');
    setSelectedOption(null);
  };

  return (
    <div className="flex flex-col w-full max-w-3xl mx-auto p-4 relative">
      {/* HITL Options Container (Claude-style floating above input) */}
      {clarificationData?.status === 'requires_clarification' && (
        <div className="mb-3 animate-fade-in-up">
          <p className="text-sm text-gray-600 mb-2 font-medium">
            {clarificationData.message}
          </p>
          <div className="flex flex-wrap gap-2">
            {clarificationData.options.map((option, idx) => (
              <button
                key={idx}
                onClick={() => handleOptionSelect(option)}
                className={`px-4 py-2 text-sm rounded-full transition-all duration-200 border 
                  ${selectedOption === option 
                    ? 'bg-purple-600 text-white border-purple-600 shadow-md' 
                    : 'bg-white text-gray-700 border-gray-300 hover:border-purple-400 hover:bg-purple-50 opacity-90'
                  }
                  ${manualInput.length > 0 ? 'opacity-50 cursor-not-allowed' : ''}
                `}
                disabled={manualInput.length > 0}
              >
                {option}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Main Input Area */}
      <form 
        onSubmit={handleSubmit} 
        className={`relative flex items-center bg-white border rounded-2xl shadow-sm transition-all
          ${selectedOption ? 'border-gray-200 bg-gray-50' : 'border-gray-300 focus-within:border-purple-500 focus-within:ring-1 focus-within:ring-purple-500'}`}
      >
        <input
          type="text"
          value={selectedOption ? `Selected: ${selectedOption}` : manualInput}
          onChange={handleTextChange}
          placeholder="Ask anything about your documents..."
          disabled={!!selectedOption}
          className="flex-grow py-3 px-4 bg-transparent outline-none text-gray-800 disabled:text-purple-700 disabled:font-medium rounded-2xl"
        />
        <button
          type="submit"
          disabled={!manualInput.trim() && !selectedOption}
          className="mr-2 p-2 rounded-xl bg-purple-500 text-white disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-purple-600 transition-colors"
        >
          {/* Send Icon */}
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
          </svg>
        </button>
      </form>
    </div>
  );
};

export default ChatInputWithFeedback;