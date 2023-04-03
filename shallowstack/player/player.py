from typing import List, TYPE_CHECKING
from shallowstack.game.action import ActionType, Action

if TYPE_CHECKING:
    from shallowstack.state_manager import GameState
    from shallowstack.poker.card import Card


class Player(object):
    def __init__(self, name: str, player_index: int, chips: int = 1000):
        self.hand: List[Card] = []
        self.name = name
        self.chips = chips
        self.went_all_in = False

    def get_action(self, game_state: "GameState") -> Action:
        """
        This is where the player must make a decision
        """
        return Action(ActionType.FOLD)

    def add_chips(self, amount: int):
        self.chips += amount

    def receive_cards(self, hand: List["Card"]):
        """
        Gives the player a hand of cards
        """
        self.hand = hand

    def can_afford_bet(self, amount: int) -> bool:
        """
        Simple check to see if bet is allowed
        """
        return self.chips >= amount

    def bet_chips(self, amount: int):
        """
        Performs the bet amount
        """
        self.chips -= amount

    def prepare_for_new_round(self):
        """
        Allows player types to cleanup internal state
        before a new round
        """
        self.went_all_in = False
        # TODO: fix this for later
        self.chips = 1000
