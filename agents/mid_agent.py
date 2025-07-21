import pandas as pd
from agents.base_agent import BaseAgent
from memory.semantic_memory import SemanticMemory
import google.generativeai as genai
import os
import re

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class MidTermAgent(BaseAgent):
    """
    Agent focusing on mid-term data to make trading decisions, using an LLM for analysis.
    """
    def __init__(self, name: str, config: dict, semantic_memory: SemanticMemory):
        super().__init__(name, config, semantic_memory)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def vote(self, memory_snapshot: dict) -> tuple[str, float]:
        """
        Analyzes mid-term memory using an LLM to decide on a trading action.
        """
        mid_term_data = memory_snapshot.get('mid_term')
        if mid_term_data is None or mid_term_data.empty or len(mid_term_data) < 20:
            return 'HOLD', 0.5
            
        ticker = mid_term_data['ticker'].iloc[-1]

        prompt = f"You are a mid-term trend analyst specializing in {ticker}. Based on the following price data and technical indicators, what is your recommendation? Provide your answer as 'VOTE: [BUY/SELL/HOLD], CONFIDENCE: [0.0-1.0]'.\n\n"
        
        # mid-term price trend with moving averages and RSI
        prompt += f"Mid-Term Price & RSI Data for {ticker} (last 20 data points):\n"
        prompt += mid_term_data[['close', 'rsi']].tail(20).to_string() + "\n\n"
        prompt += "5-day Moving Average:\n"
        prompt += mid_term_data['close'].rolling(window=5).mean().tail().to_string() + "\n\n"
        prompt += "20-day Moving Average:\n"
        prompt += mid_term_data['close'].rolling(window=20).mean().tail().to_string() + "\n"

        # Get LLM Response 
        try:
            response = self.model.generate_content(prompt)
            # Parse the Response 
            vote_match = re.search(r"VOTE:\s*(BUY|SELL|HOLD)", response.text, re.IGNORECASE)
            confidence_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", response.text, re.IGNORECASE)

            if vote_match and confidence_match:
                vote = vote_match.group(1).upper()
                confidence = float(confidence_match.group(1))
                print(f"MidTermAgent LLM Vote: {vote}, Confidence: {confidence}")
                return vote, confidence
            else:
                print(f"MidTermAgent: Could not parse LLM response: {response.text}")
                return 'HOLD', 0.5
                
        except Exception as e:
            print(f"An error occurred while calling the Gemini API: {e}")
            return 'HOLD', 0.5
