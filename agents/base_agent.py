from abc import ABC, abstractmethod
import os
import re
import pandas as pd
from google import genai
from google.genai import errors, types
from memory.semantic_memory import SemanticMemory

# The whole reply must be a single vote; prefixes, extra text and a second vote are rejected.
VOTE_PATTERN = re.compile(r"\s*VOTE:\s*(BUY|SELL|HOLD)\s*,\s*CONFIDENCE:\s*(\d+(?:\.\d*)?|\.\d+)\s*", re.IGNORECASE)


class AgentVoteError(RuntimeError):
    """An LLM agent could not produce a valid vote; the backtest stops instead of counting a HOLD."""


def gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise AgentVoteError("GEMINI_API_KEY is not set; the LLM agents cannot vote")
    return key

class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.
    """
    def __init__(self, name: str, config: dict, semantic_memory: SemanticMemory):
        """
        Initializes the agent.
        
        :param name: The name of the agent (e.g., "Short-Term Agent").
        :param config: A configuration dictionary.
        :param semantic_memory: An instance of SemanticMemory for searching textual data.
        """
        self.name = name
        self.config = config
        self.semantic_memory = semantic_memory

    @abstractmethod
    def vote(self, memory_snapshot: dict) -> tuple[str, float]:
        """
        Analyzes the given memory snapshot and returns a trading decision and confidence score.
        
        :param memory_snapshot: A dictionary containing the 'short_term', 'mid_term',   
                                and 'long_term' memory DataFrames.
        :return: A tuple containing the vote ('BUY', 'SELL', 'HOLD') and a confidence score (0.0 to 1.0).
        """
        pass

    def ask_llm(self, prompt: str) -> tuple[str, float]:
        """
        Sends the prompt to Gemini once and returns the parsed vote.

        Any provider, timeout, blocked, truncated or malformed reply raises AgentVoteError.
        A new client per call keeps cleanup local; it is closed on success and failure.
        """
        key = gemini_api_key()
        try:
            # ponytail: per-call client keeps cleanup local; share it if connection setup becomes material.
            with genai.Client(api_key=key, vertexai=False, enterprise=False,
                              http_options=types.HttpOptions(timeout=self.config['timeout_ms'],
                                                             retry_options=types.HttpRetryOptions(attempts=1))) as client:
                response = client.models.generate_content(model=self.config['model'], contents=prompt,
                                                          config={'automatic_function_calling': {'disable': True}})
        except Exception as error:
            # Provider errors can contain request bodies; keep them out of the public error.
            status = f" {error.code}" if isinstance(error, errors.APIError) else ""
            raise AgentVoteError(f"{self.name}: Gemini call failed ({type(error).__name__}{status})") from None

        candidates = response.candidates or []
        if len(candidates) != 1 or candidates[0].finish_reason != types.FinishReason.STOP or response.text is None:
            raise AgentVoteError(f"{self.name}: Gemini returned no complete answer (blocked, truncated or empty)")
        match = VOTE_PATTERN.fullmatch(response.text)
        if match is None or not 0.0 <= float(match.group(2)) <= 1.0:
            raise AgentVoteError(f"{self.name}: Gemini reply is not exactly 'VOTE: BUY|SELL|HOLD, CONFIDENCE: 0.0-1.0'")
        vote, confidence = match.group(1).upper(), float(match.group(2))
        print(f"{type(self).__name__} LLM Vote: {vote}, Confidence: {confidence}")
        return vote, confidence

if __name__ == '__main__':
    # This is an abstract class and cannot be instantiated directly.
    # The following code is for demonstration purposes of how a subclass would work.
    
    class DummyAgent(BaseAgent):
        def vote(self, memory_snapshot: dict) -> tuple[str, float]:
            print(f"Agent '{self.name}' is voting...")
            
            # Example logic: if the latest price in short-term memory is higher than the first, buy.
            short_term_data = memory_snapshot.get('short_term')
            if short_term_data is not None and not short_term_data.empty:
                if short_term_data['close'].iloc[-1] > short_term_data['close'].iloc[0]:
                    return 'BUY', 0.75
            
            return 'HOLD', 0.5

    # Example instantiation of a subclass
    dummy_config = {'some_param': 'value'}
    semantic_memory = SemanticMemory() # Dummy instance for demonstration
    agent = DummyAgent(name="Dummy Agent", config=dummy_config, semantic_memory=semantic_memory)
    
    # Create a dummy memory snapshot
    dummy_memory = {
        'short_term': pd.DataFrame({
            'close': [100, 102, 101, 103]
        }),
        'mid_term': pd.DataFrame(),
        'long_term': pd.DataFrame()
    }
    
    decision, confidence = agent.vote(dummy_memory)
    print(f"Decision: {decision}, Confidence: {confidence}") 