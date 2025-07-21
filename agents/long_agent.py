import pandas as pd
from agents.base_agent import BaseAgent
from memory.semantic_memory import SemanticMemory
import google.generativeai as genai
import os
import re

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class LongTermAgent(BaseAgent):
    """
    Agent focusing on long-term data and macroeconomic trends, using an LLM for analysis.
    """
    def __init__(self, name: str, config: dict, semantic_memory: SemanticMemory):
        super().__init__(name, config, semantic_memory)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def vote(self, memory_snapshot: dict) -> tuple[str, float]:
        """
        Analyzes long-term memory and semantic context using an LLM to decide on a trading action.
        """
        long_term_data = memory_snapshot.get('long_term')
        if long_term_data is None or long_term_data.empty:
            return 'HOLD', 0.5
            
        ticker = long_term_data['ticker'].iloc[-1]

        # Prepare the Prompt
        prompt = f"You are a long-term trading analyst specializing in {ticker}. Based on the following data, what is your recommendation? Provide your answer as 'VOTE: [BUY/SELL/HOLD], CONFIDENCE: [0.0-1.0]'.\n\n"
        
        # Add long-term price trend
        prompt += f"Long-Term Price & Indicator Data for {ticker} (last 10 data points):\n"
        prompt += long_term_data[['close', 'rsi', 'macd', 'upper_band', 'lower_band']].tail(10).to_string() + "\n\n"

        # Add semantic memory context
        try:
            semantic_results = self.semantic_memory.search_memory("market sentiment", k=3)
            if semantic_results:
                prompt += "Recent News & Reflections:\n"
                for result in semantic_results:
                    prompt += f"- {result['text']} (distance: {result['distance']:.2f})\n"
        except (IndexError, ValueError):
            # Not enough memories to search or other value error
            prompt += "No significant news or reflections found.\n"
        
        # Get LLM Response 
        try:
            response = self.model.generate_content(prompt)
            # Parse the Response 
            vote_match = re.search(r"VOTE:\s*(BUY|SELL|HOLD)", response.text, re.IGNORECASE)
            confidence_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", response.text, re.IGNORECASE)

            if vote_match and confidence_match:
                vote = vote_match.group(1).upper()
                confidence = float(confidence_match.group(1))
                print(f"LongTermAgent LLM Vote: {vote}, Confidence: {confidence}")
                return vote, confidence
            else:
                print(f"LongTermAgent: Could not parse LLM response: {response.text}")
                return 'HOLD', 0.5
                
        except Exception as e:
            print(f"An error occurred while calling the Gemini API: {e}")
            return 'HOLD', 0.5
