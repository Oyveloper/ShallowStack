from typing import List
import numpy as np
from shallowstack.game.action import AGENT_ACTIONS, Action, ActionType
from shallowstack.state_manager import GameState, PokerGameStage
from shallowstack.player.player import Player
from shallowstack.poker.card import hole_pair_idx_from_hand, hole_pair_idx_from_ids
from shallowstack.subtree.subtree_manager import SubtreeManager


class ResolvePlayer(Player):
    def __init__(self, name: str, chips: int = 1000):
        super().__init__(name, chips)
        self.r1 = np.ones(1326) / 1326
        self.r2 = np.ones(1326) / 1326

    def get_action(self, game_state: GameState) -> Action:
        end_stage = game_state.stage
        return self.resolve(
            game_state, self.r1, self.r2, end_stage, 5, 5
        )

    def resolve(
        self,
        state: GameState,
        r1: np.ndarray,
        r2: np.ndarray,
        end_stage: PokerGameStage,
        end_depth: int,
        nbr_rollouts: int,
    ) -> Action:
        strategy = np.ones((1326, len(AGENT_ACTIONS)))
        strategy /= strategy.sum(axis=1, keepdims=True)
        tree = SubtreeManager(state, end_stage, end_depth, strategy)

        strategies = np.zeros((nbr_rollouts, 1326, len(AGENT_ACTIONS)))
        for t in range(nbr_rollouts):
            v1, v2 = tree.subtree_traversal_rollout(tree.root, r1, r2)
            strategies[t] = tree.update_strategy_at_node(tree.root)
        mean_strategy = strategies.mean(axis=0)
        action_probs = mean_strategy[hole_pair_idx_from_hand(self.hand)]

        print(action_probs)

        action_index = np.random.choice(len(AGENT_ACTIONS), p=action_probs)

        return AGENT_ACTIONS[action_index]
