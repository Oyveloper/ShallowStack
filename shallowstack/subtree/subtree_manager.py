from enum import Enum
import random
from typing import List, Tuple

import numpy as np
from shallowstack.config.config import POKER_CONFIG, RESOLVER_CONFIG

from shallowstack.game.action import AGENT_ACTIONS, Action, agent_action_index
from shallowstack.neural_net.neural_net_manager import NNManager
from shallowstack.neural_net.util import create_input_vector
from shallowstack.poker.card import (
    Card,
    HOLE_PAIR_INDICES,
    hole_pair_idx_from_ids,
)
from shallowstack.poker.poker_oracle import PokerOracle
from shallowstack.state_manager import GameState, PokerGameStage
from shallowstack.state_manager.state_manager import PokerGameStateType, StateManager

NBR_EVENTS = RESOLVER_CONFIG.getint("NBR_RANDOM_EVENTS")
AVG_POT_SIZE = POKER_CONFIG.getint("AVG_POT_SIZE")


class NodeType(Enum):
    SHOWDOWN = 0
    TERMINAL = 1
    CHANCE = 2
    PLAYER = 3
    WON = 4


class SubtreeNode:
    def __init__(
        self,
        stage: PokerGameStage,
        state: GameState,
        depth: int,
        node_type: NodeType,
        strategy: np.ndarray,
        utility_matrix: np.ndarray,
        regrets: np.ndarray,
        values: np.ndarray,
    ) -> None:
        self.stage = stage
        self.state = state
        self.depth = depth
        self.node_type = node_type
        self.strategy = strategy
        self.children: List[Tuple[Action, SubtreeNode]] = []
        self.utility_matrix = utility_matrix
        self.regrets = regrets
        self.values = values

    def __str__(self, level=0, action=None) -> str:
        res = "\t" * level + f"{action} -> " + f"{self.node_type}\n"
        for action, child in self.children:
            act = action.action_type if action is not None else None

            res += child.__str__(level + 1, act)
        return res


class SubtreeManager:
    """
    Responsible for handling all actions related to generating
    the search tree that Resolve will use

    This only uses public information to generate the tree,
    as private information is irellevant
    """

    def __init__(
        self,
        state: GameState,
        end_stage: PokerGameStage,
        end_depth: int,
        strategy: np.ndarray,
    ):
        """
        Generates the initial subtree for a given game state

        This assumes that the game is in a state where a player
        should take actions

        -------
        args:
        state: The game state to generate the subtree for
        r1: The range vector for player 1
        r2: The range vector for player 2
        end_stage: The stage at which the tree should
        end_depth: The depth at which the tree should end
        strategy: The current strategy for the starting node
        """
        utility_matrix = PokerOracle.calculate_utility_matrix(state.public_cards)
        self.root = SubtreeNode(
            state.stage,
            state,
            0,
            NodeType.PLAYER,
            strategy,
            utility_matrix,
            np.zeros((1326, len(AGENT_ACTIONS))),
            np.zeros((2, 1326)),
        )
        self.end_stage = end_stage
        self.end_depth = end_depth
        self.root_player_index = state.current_player_index

        self.generate_initial_sub_tree(self.root)

        # initialize the neural net module

        self.nn_manager = NNManager()

    def generate_initial_sub_tree(self, node: SubtreeNode):
        """
        Generates the initial subtree from a given node
        """
        self.generate_children(node)

    def generate_children(self, node: SubtreeNode, action_limit: int = -1):
        """
        Adds children to the given node based on its state
        relies on the StateManager to generate the legal state/action pairs

        For chance nodes there is no action, but child states based on random deals
        """

        if node.node_type in [NodeType.SHOWDOWN, NodeType.TERMINAL, NodeType.WON]:
            return

        child_states: List[Tuple[Action, GameState]] = StateManager.get_child_states(
            node.state, NBR_EVENTS
        )

        random.shuffle(child_states)

        nbr_actions = 0
        for action, new_state in child_states:
            # Limit child generation
            if action is not None and action_limit != -1:
                if nbr_actions >= action_limit:
                    break
                nbr_actions += 1

            depth = node.depth + 1 if node.stage == new_state.stage else 0
            node_type = NodeType.PLAYER
            utility_matrix = node.utility_matrix.copy()

            if action is not None and action in [a for a, _ in node.children]:
                # Child state is already added
                continue

            if new_state.stage == PokerGameStage.SHOWDOWN:
                node_type = NodeType.SHOWDOWN
            elif new_state.game_state_type == PokerGameStateType.WINNER:
                node_type = NodeType.WON
            elif new_state.stage.value > self.end_stage.value or (
                new_state.stage == self.end_stage and depth == self.end_depth
            ):
                node_type = NodeType.TERMINAL
                utility_matrix = PokerOracle.calculate_utility_matrix(
                    new_state.public_cards
                )
            elif new_state.game_state_type == PokerGameStateType.DEALER:
                node_type = NodeType.CHANCE
                utility_matrix = PokerOracle.calculate_utility_matrix(
                    new_state.public_cards
                )

            new_node = SubtreeNode(
                new_state.stage,
                new_state,
                depth,
                node_type,
                node.strategy,
                utility_matrix,
                node.regrets.copy(),
                node.values.copy(),
            )
            node.children.append((action, new_node))

    def subtree_traversal_rollout(
        self, node: SubtreeNode, r1: np.ndarray, r2: np.ndarray, clean: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs a rollout from a given node in the subtree
        """
        # Initialize vectors to be overridden
        v1, v2 = np.zeros_like(r1), np.zeros_like(r2)
        match node.node_type:
            case NodeType.SHOWDOWN:
                v1 = node.utility_matrix @ r2.T
                v2 = -r1 @ node.utility_matrix

                v1 *= node.state.pot / AVG_POT_SIZE
                v2 *= node.state.pot / AVG_POT_SIZE

            case NodeType.WON:
                if node.state.winner_index == node.state.current_player_index:
                    v1 = np.ones_like(v1)
                    v2 = -np.ones_like(v2)
                else:
                    v1 = -1 * np.ones_like(v1)
                    v2 = np.ones_like(v2)

                v1 *= node.state.pot / AVG_POT_SIZE
                v2 *= node.state.pot / AVG_POT_SIZE

            case NodeType.TERMINAL:
                network = self.nn_manager.get_network(node.state.stage)
                in_vector = create_input_vector(
                    r1, r2, node.state.public_cards, node.state.pot
                )
                v1, v2 = network.predict_values(in_vector)
            case NodeType.PLAYER:
                ranges = [r1, r2]

                player_index = (
                    node.state.current_player_index + self.root_player_index
                ) % 2

                r_p = ranges[player_index]
                r_o = ranges[1 - player_index]

                # Rollouts generate the tree each time
                nbr_actions = RESOLVER_CONFIG.getint("NBR_ACTIONS_IN_ROLLOUT")
                self.generate_children(node, action_limit=nbr_actions)
                for action, child in node.children[-nbr_actions:]:
                    a = agent_action_index(action)
                    r_p_a = SubtreeManager.bayesian_range_update(r_p, node.strategy, a)
                    r_o_a = r_o

                    action_ranges = [r_p_a, r_o_a]
                    r1_a = action_ranges[player_index]
                    r2_a = action_ranges[1 - player_index]

                    v1_a, v2_a = self.subtree_traversal_rollout(
                        child,
                        r1_a,
                        r2_a,
                    )
                    for h in HOLE_PAIR_INDICES:
                        v1[h] += node.strategy[h, a] * v1_a[h]
                        v2[h] += node.strategy[h, a] * v2_a[h]

            case NodeType.CHANCE:
                self.generate_children(node)
                S = len(node.children)
                for _, child in node.children:
                    r1_e, r2_e = r1, r2
                    r1_e = SubtreeManager.update_range_from_public_cards(
                        r1_e, node.state.public_cards
                    )
                    r2_e = SubtreeManager.update_range_from_public_cards(
                        r2_e, node.state.public_cards
                    )

                    v1_e, v2_e = self.subtree_traversal_rollout(child, r1_e, r2_e)
                    v1 += v1_e
                    v2 += v2_e

                v1 = v1 / S
                v2 = v2 / S

        node.values = np.array([v1, v2])

        return v1, v2

    def update_strategy_at_node(self, node: SubtreeNode):
        for _, child in node.children:
            self.update_strategy_at_node(child)
        if node.node_type == NodeType.PLAYER:
            R_t = node.regrets
            player_index = (
                node.state.current_player_index + self.root_player_index
            ) % 2
            for h in HOLE_PAIR_INDICES:
                node_value = node.values[player_index][h]
                for action, child in node.children:
                    a = agent_action_index(action)
                    child_value = child.values[player_index][h]
                    R_t[h, a] += child_value - node_value
            node.regrets = R_t
            R_plus = np.clip(R_t, 0, None)
            R_plus_sum = np.sum(R_plus, axis=1)

            divisor = R_plus_sum[:, None]

            # Adding this to avoid the division by 0
            divisor[np.where(divisor == 0)] = 1 / R_t.shape[1]

            node.strategy = R_plus / divisor

            return node.strategy

    @staticmethod
    def bayesian_range_update(
        range: np.ndarray, strategy: np.ndarray, action_index: int
    ):
        p_action = np.sum(strategy[:, action_index]) / np.sum(strategy) + 0.0001

        res = range * strategy[:, action_index] / p_action
        return res

    @staticmethod
    def update_range_from_public_cards(
        r: np.ndarray, new_public_cards: List[Card]
    ) -> np.ndarray:
        """
        Updates the ranges to reflect the new public cards
        """
        r = r.copy()
        for card in new_public_cards:
            c_id = card.id
            for other_card_id in range(52):
                if c_id == other_card_id:
                    continue
                hole_pair_idx = hole_pair_idx_from_ids(c_id, other_card_id)
                r[hole_pair_idx] = 0
        return r
