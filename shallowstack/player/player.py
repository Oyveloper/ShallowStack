from typing import List
from shallowstack.game.action import ActionType, Action
from shallowstack.poker.card import Card


class Player(object):
    def __init__(self, name: str, chips: int = 1000):
        self.hand: List[Card] = []
        self.name = name
        self.chips = chips

    def get_action(self, legal_actions: List[ActionType]) -> Action:
        """
        This is where the player must make a decision
        """
        return Action(ActionType.FOLD)

    def add_chips(self, amount: int):
        self.chips += amount

    def receive_cards(self, hand: List[Card]):
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
