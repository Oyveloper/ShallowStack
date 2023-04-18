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
    def __init__(
        self,
        name: str,
        resolve_probability: float = 0.5,
        range_size: int = 1326,
        show_internals: bool = False,
    ):
        super().__init__(name)

        self.range_size = range_size
        self.r1 = np.ones(range_size) / range_size
        self.r2 = np.ones(range_size) / range_size

        self.opponent_strategy = np.ones((range_size, len(AGENT_ACTIONS))) / len(
            AGENT_ACTIONS
        )

        self.resolver = Resolver()

        self.resolve_probability = resolve_probability
        self.show_internals = show_internals

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
                self.hand, len(game_state.player_bets)
            )
        else:
            win_probability = PokerOracle.hole_hand_winning_probability_rollout(
                self.hand, game_state.public_info, len(game_state.player_bets)
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
        end_depth = 0
        if current_stage.value < PokerGameStage.RIVER.value:
            end_stage = PokerGameStage((game_state.stage.value + 1))
        else:
            end_stage = current_stage
            end_depth = 10

        nbr_rollouts = RESOLVER_CONFIG.getint("NBR_ROLLOUTS")

        action, self.r1, self.r2, self.opponent_strategy = self.resolver.resolve(
            game_state,
            self.r1,
            self.r2,
            end_stage,
            end_depth,
            nbr_rollouts,
            self.show_internals,
        )

        return action

    def prepare_for_new_round(self):
        super().prepare_for_new_round()
        self.r1 = np.ones(self.range_size) / self.range_size
        self.r2 = np.ones(self.range_size) / self.range_size

    def inform_of_action(self, action: Action, player: Player):
        if player.name == self.name:
            return

        action_type = action.action_type

        closest_amount = action.amount
        if action_type == ActionType.RAISE:
            closest_amount = ALLOWED_RAISES[
                int(np.argmin(np.abs(action.amount - np.array(ALLOWED_RAISES))))
            ]

        action_index = AGENT_ACTIONS.index(Action(action_type, closest_amount))

        r2 = SubtreeManager.bayesian_range_update(
            self.r2, self.opponent_strategy, action_index
        )

        if np.sum(r2) == 0:
            # Must avoid putting a stupid range
            return

        self.r2 = r2
