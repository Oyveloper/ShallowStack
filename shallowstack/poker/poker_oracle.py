from typing import List

from shallowstack.poker.card import Card

from shallowstack.poker.hash import hash_quinary
from shallowstack.poker.tables import (
    BINARIES_BY_ID,
    FLUSH,
    NO_FLUSH_5,
    NO_FLUSH_6,
    NO_FLUSH_7,
    SUITBIT_BY_ID,
    SUITS,
)

MIN_CARDS = 5
MAX_CARDS = 7

NO_FLUSHES = {5: NO_FLUSH_5, 6: NO_FLUSH_6, 7: NO_FLUSH_7}


class PokerOracle:
    @staticmethod
    def calculate_utility_matrix():
        pass

    @staticmethod
    def evaluate_hand(hand: List[Card]) -> int:
        """
        External wrapper to be used for evaluating a hand of cards.
        Simplifies the conversion from the Card class to integers that is needed
        for the hashing
        """

        int_cards = [c.id for c in hand]
        return PokerOracle._evaluate_hand(int_cards)

    @staticmethod
    def _evaluate_hand(cards: List[int]) -> int:
        """
        This is where the magic happens!
        Internal funciton that uses the integer representation of the cards to
        find the precalculated value of a hand of cards.

        this is adapted from https://github.com/HenryRLee/PokerHandEvaluator
        which precomputes tables for evaluating hands of sizes 5-9
        """
        hand_size = len(cards)
        no_flush = NO_FLUSHES[hand_size]

        suit_hash = 0
        for card in cards:
            suit_hash += SUITBIT_BY_ID[card]

        flush_suit = SUITS[suit_hash] - 1

        if flush_suit != -1:
            hand_binary = 0

            for card in cards:
                if card % 4 == flush_suit:
                    hand_binary |= BINARIES_BY_ID[card]

            return FLUSH[hand_binary]

        hand_quinary = [0] * 13
        for card in cards:
            hand_quinary[card // 4] += 1

        return no_flush[hash_quinary(hand_quinary, hand_size)]
