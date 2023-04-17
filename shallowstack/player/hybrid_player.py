import numpy as np
from shallowstack.config.config import RESOLVER_CONFIG
from shallowstack.game.action import AGENT_ACTIONS, ALLOWED_RAISES, Action, ActionType
from shallowstack.poker.poker_oracle import PokerOracle
from shallowstack.resolver.resolver import Resolver
from shallowstack.state_manager import GameState, PokerGameStage
from shallowstack.player.player import Player
from shallowstack.state_manager.state_manager import StateManager
from shallowstack.subtree.subtree_manager import SubtreeManager


class HybridPlayer(Player):
    def __init__(self, name: str, chips: int = 1000, resolve_probability: float = 0.5):
        super().__init__(name, chips)

        self.r1 = np.ones(1326) / 1326
        self.r2 = np.ones(1326) / 1326

        self.opponent_strategy = np.ones((1326, len(AGENT_ACTIONS))) / len(
            AGENT_ACTIONS
        )

        self.resolver = Resolver()

        self.resolve_probability = resolve_probability

    def get_action(self, game_state: GameState) -> Action:
        """
        Handles the logic for deciding what action to take.
        """
        if np.random.random() < self.resolve_probability:
            return self.resolve_action(game_state)
        else:
            return self.rollout_action(game_state)

    def rollout_action(self, game_state: GameState) -> Action:
        """
        Handles logic for using rollout based strategy
        """
        win_probability = 0
        if game_state.stage == PokerGameStage.PRE_FLOP:
            win_probability = PokerOracle.hole_hand_winning_probability_cheat_sheet(
                self.hand, len(game_state.players)
            )
        else:
            win_probability = PokerOracle.hole_hand_winning_probability_rollout(
                self.hand, game_state.public_cards, len(game_state.players)
            )

        legal_actions = StateManager.get_legal_actions(game_state)

        if win_probability < 0.1:
            return Action(ActionType.FOLD, 0)
        elif win_probability < 0.5:
            idx = np.random.randint(len(legal_actions))
            act_type = legal_actions[idx]
            if act_type == ActionType.RAISE:
                return Action(act_type, np.random.choice(ALLOWED_RAISES))
            else:
                return Action(act_type, 0)
        elif win_probability < 0.8:
            return Action(ActionType.CALL, 0)
        else:
            return Action(ActionType.RAISE, np.random.choice(ALLOWED_RAISES))

    def resolve_action(self, game_state: GameState) -> Action:
        """
        Handles logic for using resolve based strategy
        """
        current_stage = game_state.stage
        if current_stage.value < PokerGameStage.RIVER.value:
            end_stage = current_stage
        else:
            end_stage = PokerGameStage((game_state.stage.value + 1))

        nbr_rollouts = RESOLVER_CONFIG.getint("NBR_ROLLOUTS")

        action, self.r1, self.r2, self.opponent_strategy = self.resolver.resolve(
            game_state, self.r1, self.r2, end_stage, 5, nbr_rollouts
        )

        return action

    def prepare_for_new_round(self):
        super().prepare_for_new_round()
        self.r1 = np.ones(1326) / 1326
        self.r2 = np.ones(1326) / 1326

    def inform_of_action(self, action: Action, player: Player):
        if player.name == self.name:
            return

        print("Updating oponent range!")

        action_type = action.action_type

        closest_amount = action.amount
        if action_type == ActionType.RAISE:
            closest_amount = ALLOWED_RAISES[
                int(np.argmin(np.abs(action.amount - np.array(ALLOWED_RAISES))))
            ]

        action_index = AGENT_ACTIONS.index(Action(action_type, closest_amount))

        self.r2 = SubtreeManager.bayesian_range_update(
            self.r2, self.opponent_strategy, action_index
        )
