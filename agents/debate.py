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
        Gathers votes from all agents and determines the final decision.
        """
        votes = []
        for agent in self.agents:
            decision, confidence = agent.vote(memory_snapshot)
            votes.append({'agent': agent.name, 'decision': decision, 'confidence': confidence})

        final_decision, final_confidence = self._resolve_votes(votes)
        
        return final_decision, final_confidence, votes

    def _resolve_votes(self, votes: List[dict]) -> Tuple[str, float]:
        """
        Resolves the collected votes into a single decision.
        This can be replaced with a more sophisticated method.
        """
        if not votes:
            return 'HOLD', 0.5

        # Simple weighted average of votes
        buy_confidence = sum(v['confidence'] for v in votes if v['decision'] == 'BUY')
        sell_confidence = sum(v['confidence'] for v in votes if v['decision'] == 'SELL')
        hold_confidence = sum(v['confidence'] for v in votes if v['decision'] == 'HOLD')

        if buy_confidence > sell_confidence and buy_confidence > hold_confidence:
            return 'BUY', buy_confidence / len(votes)
        elif sell_confidence > buy_confidence and sell_confidence > hold_confidence:
            return 'SELL', sell_confidence / len(votes)
        else:
            return 'HOLD', hold_confidence / len(votes)
