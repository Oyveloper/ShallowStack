import numpy as np
from shallowstack.config.config import RESOLVER_CONFIG
from shallowstack.game.action import AGENT_ACTIONS, ALLOWED_RAISES, Action, ActionType
from shallowstack.resolver.resolver import Resolver
from shallowstack.state_manager import GameState, PokerGameStage
from shallowstack.player.player import Player
from shallowstack.subtree.subtree_manager import SubtreeManager


class ResolvePlayer(Player):
    def __init__(self, name: str, chips: int = 1000):
        super().__init__(name, chips)
        self.r1 = np.ones(1326) / 1326
        self.r2 = np.ones(1326) / 1326

        self.opponent_strategy = np.ones((1326, len(AGENT_ACTIONS))) / len(
            AGENT_ACTIONS
        )

        self.resolver = Resolver()

    def get_action(self, game_state: GameState) -> Action:
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
