from typing import List, Tuple
from shallowstack.game.action import Action, ActionType
from shallowstack.game.poker_game import GameManager, GameState


class NumericalGameState:
    def __init__(self, state: GameState):
        self.state = state


ALLOWED_RAISES = [
    5,
    10,
]


class StateManager:
    @staticmethod
    def get_actions_with_new_states(
        state: GameState,
    ) -> List[Tuple[ActionType, GameState]]:
        game_state = state.copy()
        action_types = GameManager.get_legal_actions(game_state)

        result = []
        for action_type in action_types:

            # Generalize the generation so that we can have a fixed
            # amount of allowed raises exposed to re-solvers
            amounts = [0]
            if action_type == ActionType.RAISE:
                amounts = ALLOWED_RAISES

            for amount in amounts:
                action = Action(action_type, amount)
                new_state = GameManager.apply_action(game_state, action)

                result.append((action, new_state))

        return []
