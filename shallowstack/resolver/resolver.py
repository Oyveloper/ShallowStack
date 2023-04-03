from typing import Tuple
import numpy as np
from shallowstack.game.action import AGENT_ACTIONS, Action

from shallowstack.state_manager.state_manager import GameState, PokerGameStage
from shallowstack.subtree.subtree_manager import SubtreeManager


class Resolver:
    def resolve(
        self,
        state: GameState,
        r1: np.ndarray,
        r2: np.ndarray,
        end_stage: PokerGameStage,
        end_depth: int,
        nbr_rollouts: int,
    ) -> Tuple[Action, np.ndarray, np.ndarray]:
        """
        Runs the Re-Solve algorithm to generate an optimal action, and new ranges
        """
        strategy = np.ones((1326, len(AGENT_ACTIONS)))
        strategy /= strategy.sum(axis=1, keepdims=True)
        tree = SubtreeManager(state, end_stage, end_depth, strategy)

        r1 = r1.copy()
        r2 = r2.copy()

        strategies = np.zeros((nbr_rollouts, 1326, len(AGENT_ACTIONS)))
        for t in range(nbr_rollouts):
            tree.subtree_traversal_rollout(tree.root, r1, r2)
            strategies[t] = tree.update_strategy_at_node(tree.root)

        mean_strategy = strategies.mean(axis=0)

        action_probs = r1 @ mean_strategy
        action_probs /= np.sum(action_probs)

        action_index = np.random.choice(len(AGENT_ACTIONS), p=action_probs)

        r1 = SubtreeManager.bayesian_range_update(r1, mean_strategy, action_index)

        print(action_probs)

        return (AGENT_ACTIONS[action_index], r1, r2)
