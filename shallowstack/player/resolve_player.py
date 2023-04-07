import numpy as np
from shallowstack.config.config import RESOLVER_CONFIG
from shallowstack.game.action import Action
from shallowstack.resolver.resolver import Resolver
from shallowstack.state_manager import GameState, PokerGameStage
from shallowstack.player.player import Player


class ResolvePlayer(Player):
    def __init__(self, name: str, chips: int = 1000):
        super().__init__(name, chips)
        self.r1 = np.ones(1326) / 1326
        self.r2 = np.ones(1326) / 1326

        self.resolver = Resolver()

    def get_action(self, game_state: GameState) -> Action:
        current_stage = game_state.stage
        if current_stage.value < PokerGameStage.RIVER.value:
            end_stage = current_stage
        else:
            end_stage = PokerGameStage((game_state.stage.value + 1))

        nbr_rollouts = RESOLVER_CONFIG.getint("NBR_ROLLOUTS")

        action, r1, r2 = self.resolver.resolve(
            game_state, self.r1, self.r2, end_stage, 5, nbr_rollouts
        )

        self.r1 = r1
        self.r2 = r2

        return action

    def prepare_for_new_round(self):
        super().prepare_for_new_round()
        self.r1 = np.ones(1326) / 1326
        self.r2 = np.ones(1326) / 1326
