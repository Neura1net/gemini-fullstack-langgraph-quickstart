#!/usr/bin/env python3
"""
ask_with_ollama.py

This script re-uses the LangGraph-based ResearchAgent defined in 
google-gemini/gemini-fullstack-langgraph-quickstart (backend/src/agent/graph.py)
but swaps out all Gemini API calls for a local Ollama (DeepSeek-R1:8B) client.

Usage:
    1. Ensure Ollama is running (`ollama serve`) and the model is downloaded:
         ollama run deepseek-r1:8b 
    2. From this script’s directory, install dependencies for the Quickstart backend:
         cd backend
         pip install .
       (Make sure LangGraph, FastAPI, etc. are installed as per the README.) 
    3. Run this script:
         python3 ask_with_ollama.py
    4. Type questions at the prompt; the agent will iterate (web search + reasoning)
       and return a final answer with citations.
"""

import ollama  # pip install ollama
import os
import asyncio

# Point to the local LangGraph backend (adjust this path if needed)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "backend/src"))

# Import the ResearchAgent from the Quickstart repo
from agent.graph import ResearchAgent, LLMSpecification


class OllamaLLM:
    """
    A simple wrapper around Ollama’s Python API that matches the interface
    expected by the Quickstart’s ResearchAgent.
    """

    def __init__(self, model_name: str = "deepseek-r1:8b"):
        self.model_name = model_name

    async def generate(self, prompt: str) -> str:
        """
        Call ollama.generate asynchronously and return the “response” text.
        """
        # Ollama’s Python client is synchronous; wrap it in an executor
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: ollama.generate(model=self.model_name, prompt=prompt)
        )
        return result["response"]


async def main():
    # 1. Initialize an OllamaLLM instance
    ollama_llm = OllamaLLM(model_name="deepseek-r1:8b")

    # 2. Create an LLMSpecification that tells the agent how to call our LLM.
    #    The Quickstart’s graph.py expects an LLMSpecification with at least:
    #      - a “generate_fn” that takes a prompt and returns text,
    #      - a “max_tokens” limit (optional here).
    llm_spec = LLMSpecification(
        name="deepseek-r1:8b (local)",
        generate_fn=ollama_llm.generate,
        max_tokens=2048  # adjust as needed
    )

    # 3. Instantiate the ResearchAgent, passing in our Ollama-based LLM spec.
    #    By default, ResearchAgent will:
    #      - generate initial web-search queries,
    #      - call the LLM for search-refinement and answer synthesis,
    #      - make Google Search API calls (you still need a valid API key in .env).
    agent = ResearchAgent(llm_spec=llm_spec)

    print("Ask (type ‘exit’ or Ctrl+C to quit):")
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                break

            # 4. Run the agent’s ‘run’ coroutine, which does the multi-step research loop.
            answer = await agent.run(user_input)

            # 5. Print out the final answer (including citations).
            print("\n=== Agent’s Answer ===\n")
            print(answer)
            print("\n======================\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    # Launch the asyncio event loop
    asyncio.run(main())
