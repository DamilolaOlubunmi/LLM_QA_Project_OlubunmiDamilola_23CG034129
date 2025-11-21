#!/usr/bin/env python3
"""
LLM Question and Answering CLI Application
Name: Olubunmi Damilola
Matric No: 23CG034129
"""

import os
import string
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

# Load environment variables
load_dotenv()


def preprocess_question(question):
    """
    Preprocess the input question:
    - Lowercasing
    - Tokenization
    - Punctuation removal
    """
    original = question
    question_lower = question.lower()
    question_no_punct = question_lower.translate(str.maketrans('', '', string.punctuation))
    tokens = question_no_punct.split()
    processed = ' '.join(tokens)

    print("--- Preprocessing Steps ---")
    print(f"Original: {original}")
    print(f"Lowercased: {question_lower}")
    print(f"Punctuation Removed: {question_no_punct}")
    print(f"Tokens: {tokens}")
    print(f"Processed: {processed}")
    print("-------------------------")

    return processed, original


def query_llm(question, api_key):
    """
    Send question to Google Gemini LLM API
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Answer the following question concisely: {question}"

        print(f"Sending to LLM API (Model: {model.model_name})...")

        response = model.generate_content(prompt)

        if hasattr(response, "text"):
            return response.text
        else:
            return "No response generated. Please try again."

    except GoogleAPIError as e:
        return f"API Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    print("=" * 60)
    print("LLM Question and Answering CLI")
    print("Name:   Olubunmi Damilola")
    print("Matric No: 23CG034129")
    print("=" * 60)

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("\nERROR: GEMINI_API_KEY not found!")
        print("Please create a .env file with:")
        print("GEMINI_API_KEY=your_api_key_here")
        print("\nGet your key from: https://makersuite.google.com/app/apikey")
        return

    print("\nAPI Key loaded successfully")
    print("\nType 'quit' or 'exit' to close the application.\n")

    while True:
        print("-" * 60)
        question = input("Enter your question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the LLM Q&A CLI. Goodbye!")
            break

        if not question:
            print("Please enter a valid question.\n")
            continue

        processed_question, original_question = preprocess_question(question)

        print("Processing your question...")
        answer = query_llm(original_question, api_key)

        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()