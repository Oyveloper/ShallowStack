from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
from shallowstack.game.action import AGENT_ACTIONS, Action, agent_action_index
from shallowstack.state_manager import GameState, PokerGameStage
from shallowstack.poker.card import HOLE_PAIR_INDICES, Card, Deck
from shallowstack.poker.poker_oracle import PokerOracle
from shallowstack.state_manager.state_manager import PokerGameStateType, StateManager

NBR_EVENTS = 5


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
    ) -> None:
        self.stage = stage
        self.state = state
        self.depth = depth
        self.node_type = node_type
        self.strategy = strategy
        self.children: List[Tuple[Action, SubtreeNode]] = []
        self.utility_matrix = utility_matrix
        self.regrets = regrets


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
        )
        self.end_stage = end_stage
        self.end_depth = end_depth

        self.generate_initial_sub_tree(self.root)

    def generate_initial_sub_tree(self, node: SubtreeNode):
        """
        Generates the initial subtree from a given node
        """
        self.generate_children(node)
        for (_, child) in node.children:
            self.generate_initial_sub_tree(child)

    def generate_children(self, node: SubtreeNode):
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

        for (action, new_state) in child_states:
            depth = node.depth + 1 if node.stage == new_state.stage else 0
            node_type = NodeType.PLAYER
            utility_matrix = node.utility_matrix.copy()

            if new_state.stage == PokerGameStage.SHOWDOWN:
                node_type = NodeType.SHOWDOWN
            elif new_state.game_state_type == PokerGameStateType.WINNER:
                node_type = NodeType.WON
            elif new_state.stage == self.end_stage and depth == self.end_depth:
                node_type = NodeType.TERMINAL
            elif new_state.stage_finished:
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
            )
            node.children.append((action, new_node))

    def refresh_chance_node(
        self,
        node: SubtreeNode,
        utility_matrix: Optional[np.ndarray] = None,
    ):
        """
        Updates a chance node in the already generated seach tree

        It will draw new events for the chance node, and update utility matrix of itself and children
        """
        if utility_matrix is None and node.node_type == NodeType.CHANCE:
            # Starting update from this node
            generated_states = StateManager.get_child_states(node.state, NBR_EVENTS)
            for i, (_, child) in enumerate(node.children):
                s = generated_states[i][1]
                utility_matrix = PokerOracle.calculate_utility_matrix(s.public_cards)

                child.utility_matrix = utility_matrix
                child.state = s.copy()
                self.refresh_chance_node(child, utility_matrix)
        elif utility_matrix is not None:
            for (_, child) in node.children:
                child.utility_matrix = utility_matrix
                self.refresh_chance_node(child, utility_matrix)

    def subtree_traversal_rollout(
        self, node: SubtreeNode, r1: np.ndarray, r2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs a rollout from a given node in the subtree
        """
        v1, v2 = np.zeros_like(r1), np.zeros_like(r2)
        match node.node_type:
            case NodeType.SHOWDOWN:
                v1 = node.utility_matrix @ r2.T
                v2 = -r2 @ node.utility_matrix

            case NodeType.TERMINAL:
                # TODO: evaluate this with a neural network!
                v1 = node.utility_matrix @ r2.T
                v2 = -r2 @ node.utility_matrix
            case NodeType.PLAYER:

                ranges = [r1, r2]
                player_index = node.state.current_player_index

                r_p = ranges[player_index]
                r_o = ranges[1 - player_index]
                for (action, child) in node.children:
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
                self.refresh_chance_node(node)
                S = len(node.children)
                for (_, child) in node.children:
                    r1_e, r2_e = r1, r2
                    v1_e, v2_e = self.subtree_traversal_rollout(child, r1_e, r2_e)
                    for h in HOLE_PAIR_INDICES:
                        v1[h] += v1_e[h] / S
                        v2[h] += v2_e[h] / S

        return v1, v2

    def update_strategy_at_node(self, node: SubtreeNode):
        for (_, child) in node.children:
            self.update_strategy_at_node(child)
        if node.node_type == NodeType.PLAYER:
            R_t = node.regrets
            for h in HOLE_PAIR_INDICES:
                for (action, child) in node.children:
                    a = agent_action_index(action)
                    R_t[h, a] += 1
            R_plus = np.clip(R_t, 0, None)
            R_plus_sum = np.sum(R_plus, axis=1)
            node.strategy = R_plus / R_plus_sum[:, None]
            return node.strategy

    @staticmethod
    def bayesian_range_update(
        range: np.ndarray, strategy: np.ndarray, action_index: int
    ):
        p_action = np.sum(strategy[:, action_index]) / np.sum(strategy)
        return range * strategy[:, action_index] / p_action
