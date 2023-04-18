from os import stat
from typing import List, Tuple
import numpy as np


SUIT_NUM_DICT = {"C": 0, "D": 1, "H": 2, "S": 3}
RANK_NUM_DICT = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

NUM_SUIT_DICT = {v: k for k, v in SUIT_NUM_DICT.items()}
NUM_RANK_DICT = {v: k for k, v in RANK_NUM_DICT.items()}

HOLE_PAIR_INDICES = [i for i in range(1326)]


class Card:
    def __init__(self, suit: str, rank: str):
        """
        Computes an id to be used in hashing and evaluating
        Shift the dict value down to an index which is what the
        hash expects
        """
        rank_index = RANK_NUM_DICT[rank] - 2
        self.id = rank_index * 4 + SUIT_NUM_DICT[suit]
        self.suit = suit
        self.rank = rank
        self.rank_value = RANK_NUM_DICT[rank]
        self.suit_value = SUIT_NUM_DICT[suit]
        self.value = SUIT_NUM_DICT[suit] * 14 + RANK_NUM_DICT[rank]

    def __repr__(self):
        return f"{self.rank}{self.suit}"

    @staticmethod
    def from_id(id: int) -> "Card":
        """
        Converts an id to a card
        """
        rank = NUM_RANK_DICT[id // 4 + 2]
        suit = NUM_SUIT_DICT[id % 4]
        return Card(suit, rank)

    def __eq__(self, other):
        return self.id == other.id


def hole_pair_idx_from_ids(id1: int, id2: int) -> int:
    """
    Computes the index of the hole pair from the ids of the cards
    """
    n = 52
    # Big scary formula gotten from Stack overflow
    # This calculates the linear array index from the
    # upper tiral matrix index which is what we in effect have
    assert id1 != id2
    if id1 > id2:
        id1, id2 = id2, id1
    i = id1
    j = id2
    return int((n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1)


def hole_pair_idx_from_hand(hand: List[Card]) -> int:
    """
    Helper function to wrap the more primitive function
    hole_pair_idx_from_ids
    """
    if len(hand) != 2:
        raise ValueError("Hand must have two cards")

    return hole_pair_idx_from_ids(hand[0].id, hand[1].id)


def hole_card_ids_from_pair_idx(idx: int) -> Tuple[int, int]:
    """
    Computes the ids of the cards from the hole pair index
    """
    n = 52

    # Big scary formula gotten from Stack overflow
    # Calculates the upper triangular matrix index (i, j)
    # given the linear array index
    i = int(n - 2 - np.floor(np.sqrt(-8 * idx + 4 * n * (n - 1) - 7) / 2.0 - 0.5))
    j = int(idx + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2)

    return (i, j)


class Deck:
    def __init__(self, low_card_value: int = 2, high_card_value: int = 14):
        nr_cards_per_suit = high_card_value - low_card_value + 1
        nr_cards = nr_cards_per_suit * 4

        self.low_card_value = low_card_value
        self.high_card_value = high_card_value
        self.card_distribution = np.ones(nr_cards) / nr_cards
        self.cards = [
            Card(NUM_SUIT_DICT[i], NUM_RANK_DICT[j])
            for j in range(low_card_value, high_card_value + 1)
            for i in range(4)
        ]

    def draw(self, nr_cards: int) -> List[Card]:
        # Draw random indices from the card distribution
        if np.sum(self.card_distribution) == 0:
            raise Exception("No cards left in deck")
        indices = np.random.choice(
            len(self.card_distribution),
            nr_cards,
            replace=False,
            p=self.card_distribution,
        )

        # Remove these drawn cards from the deck
        # and redistribute so that the remaining cards
        # have a uniform distribution
        self.card_distribution[indices] = 0
        s = np.sum(self.card_distribution)
        if s == 0:
            s = 1
        self.card_distribution = self.card_distribution / s

        return [self.cards[i] for i in indices]

    def copy(self):
        new_deck = Deck(self.low_card_value, self.high_card_value)
        new_deck.card_distribution = self.card_distribution.copy()
        return new_deck

    def remove_cards(self, cards: List[Card]):
        for card in cards:
            self.card_distribution[card.id] = 0
        s = np.sum(self.card_distribution)
        if s == 0:
            s = 1
        self.card_distribution = self.card_distribution / s
