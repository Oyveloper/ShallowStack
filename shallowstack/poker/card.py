from typing import List
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


class Card:
    def __init__(self, suit: str, rank: str):
        # Computes an id to be used in hashing and evaluating
        # Shift the dict value down to an index which is what the
        # hash expects
        rank_index = RANK_NUM_DICT[rank] - 2
        self.id = rank_index * 4 + SUIT_NUM_DICT[suit]
        self.suit = suit
        self.rank = rank
        self.rank_value = RANK_NUM_DICT[rank]
        self.suit_value = SUIT_NUM_DICT[suit]
        self.value = SUIT_NUM_DICT[suit] * 14 + RANK_NUM_DICT[rank]

    def __repr__(self):
        return f"{self.rank}{self.suit}"


class Deck:
    def __init__(self, low_card_value: int = 2, high_card_value: int = 14):
        nr_cards_per_suit = high_card_value - low_card_value + 1
        nr_cards = nr_cards_per_suit * 4

        self.low_card_value = low_card_value
        self.high_card_value = high_card_value
        self.card_distribution = np.ones(nr_cards) / nr_cards
        self.cards = [
            Card(NUM_SUIT_DICT[i], NUM_RANK_DICT[j])
            for i in range(4)
            for j in range(low_card_value, high_card_value + 1)
        ]

    def draw(self, nr_cards: int) -> List[Card]:
        # Draw random indices from the card distribution
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
        self.card_distribution = self.card_distribution / np.sum(self.card_distribution)

        return [self.cards[i] for i in indices]

    def copy(self):
        new_deck = Deck(self.low_card_value, self.high_card_value)
        new_deck.card_distribution = self.card_distribution.copy()
        return new_deck


# def pair_from_index(index: int) -> Tuple[Card, Card]:
#     pass
#
#
# def index_from_pair(card1: Card, card2: Card) -> int:
#     pass
