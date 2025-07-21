from typing import List, Tuple
from agents.base_agent import BaseAgent

class Debate:
    """
    Orchestrates a debate among trading agents to reach a consensus.
    """
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents

    def run(self, memory_snapshot: dict) -> Tuple[str, float, List[dict]]:
        """
        Gathers votes from all agents and determines the final decision using confidence weighting.
        """
        votes = []
        for agent in self.agents:
            decision, confidence = agent.vote(memory_snapshot)
            votes.append({'agent': agent.name, 'decision': decision, 'confidence': confidence})

        final_decision, final_confidence = self._resolve_votes_with_weighting(votes)
        
        return final_decision, final_confidence, votes

    def _resolve_votes_with_weighting(self, votes: List[dict]) -> Tuple[str, float]:
        """
        Resolves the collected votes into a single decision using confidence as a weight.
        """
        if not votes:
            return 'HOLD', 0.5

        buy_strength = sum(v['confidence'] for v in votes if v['decision'] == 'BUY')
        sell_strength = sum(v['confidence'] for v in votes if v['decision'] == 'SELL')
        
        # Normalize by the sum of all confidences to get a weighted average
        total_confidence = sum(v['confidence'] for v in votes)
        if total_confidence == 0:
            return 'HOLD', 0.0

        # The final confidence is the difference between buy and sell strength, normalized
        net_strength = buy_strength - sell_strength
        final_confidence = abs(net_strength) / total_confidence

        if net_strength > 0:
            return 'BUY', final_confidence
        elif net_strength < 0:
            return 'SELL', final_confidence
        else:
            return 'HOLD', 1.0 - final_confidence # Confidence in HOLD is inverse of conviction
