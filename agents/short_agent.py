import pandas as pd
from agents.base_agent import BaseAgent
from memory.semantic_memory import SemanticMemory
import google.generativeai as genai
import os
import re

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class ShortTermAgent(BaseAgent):
    """
    Agent focusing on short-term data to make trading decisions, using an LLM for analysis.
    """
    def __init__(self, name: str, config: dict, semantic_memory: SemanticMemory):
        super().__init__(name, config, semantic_memory)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def vote(self, memory_snapshot: dict) -> tuple[str, float]:
        """
        Analyzes short-term memory using an LLM to decide on a trading action.
        """
        short_term_data = memory_snapshot.get('short_term')
        if short_term_data is None or short_term_data.empty:
            return 'HOLD', 0.5

        # --- Prepare the Prompt ---
        prompt = "You are a short-term momentum trader. Based on the recent price action, what is your recommendation? Provide your answer as 'VOTE: [BUY/SELL/HOLD], CONFIDENCE: [0.0-1.0]'.\n\n"
        prompt += "Short-Term Price Data (last 10 data points):\n"
        prompt += short_term_data.tail(10).to_string() + "\n"

        # --- Get LLM Response ---
        try:
            response = self.model.generate_content(prompt)
            # --- Parse the Response ---
            vote_match = re.search(r"VOTE:\s*(BUY|SELL|HOLD)", response.text, re.IGNORECASE)
            confidence_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", response.text, re.IGNORECASE)

            if vote_match and confidence_match:
                vote = vote_match.group(1).upper()
                confidence = float(confidence_match.group(1))
                print(f"ShortTermAgent LLM Vote: {vote}, Confidence: {confidence}")
                return vote, confidence
            else:
                print(f"ShortTermAgent: Could not parse LLM response: {response.text}")
                return 'HOLD', 0.5
                
        except Exception as e:
            print(f"An error occurred while calling the Gemini API: {e}")
            return 'HOLD', 0.5
