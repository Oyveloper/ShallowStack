from typing import List, TYPE_CHECKING
from shallowstack.game.action import ActionType, Action

if TYPE_CHECKING:
    from shallowstack.state_manager import GameState
    from shallowstack.poker.card import Card


class Player(object):
    def __init__(self, name: str):
        self.hand: List[Card] = []
        self.name = name
        self.went_all_in = False
        self.show_internals: bool = False

    def get_action(self, game_state: "GameState") -> Action:
        """
        This is where the player must make a decision
        """
        return Action(ActionType.FOLD)

    def receive_cards(self, hand: List["Card"]):
        """
        Gives the player a hand of cards
        """
        self.hand = hand

    def prepare_for_new_round(self):
        """
        Allows player types to cleanup internal state
        before a new round
        """
        self.went_all_in = False
        # TODO: fix this for later
        self.chips = 1000

    def inform_of_action(self, action: Action, player: "Player"):
        """
        Interface that allows the game manager to inform
        players about actions that happened during the game.

        Mainly here to allow resolve player to update ranges
        """
        pass
