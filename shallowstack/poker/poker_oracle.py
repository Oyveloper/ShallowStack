import pickle
from typing import List, NamedTuple
import numpy as np

from tqdm import tqdm

from shallowstack.poker.card import (
    NUM_RANK_DICT,
    RANK_NUM_DICT,
    Card,
    Deck,
    hole_card_ids_from_pair_idx,
)

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


class PokerHandType(NamedTuple):
    c1_rank: int
    c2_rank: int
    suited: bool


class PokerOracle:
    @staticmethod
    def calculate_utility_matrix(
        public_cards: List[Card],
        range_length: int = 1326,
    ) -> np.ndarray:
        """
        Calculates the utility matrix for the given ranges

        a value of 1 at (i, j) means that hole card i wins over hole card j
        """
        hand_strenghts = np.zeros(range_length)
        for i in range(range_length):
            h1_ids = hole_card_ids_from_pair_idx(i)
            hand = [Card.from_id(h1_ids[0]), Card.from_id(h1_ids[1])]

            duplicate_cards = False
            for card in hand:
                if card in public_cards:
                    duplicate_cards = True

            if duplicate_cards:
                continue

            hand_strenghts[i] = PokerOracle.evaluate_hand(hand + public_cards)

        m = np.sign(-np.subtract.outer(hand_strenghts, hand_strenghts))
        return m

    @staticmethod
    def hand_to_hand_type(hand: List[Card]) -> PokerHandType:
        assert len(hand) == 2
        c1, c2 = hand
        if c1.rank_value > c2.rank_value:
            c1, c2 = c2, c1
        return PokerHandType(c1.rank_value, c2.rank_value, c1.suit == c2.suit)

    @staticmethod
    def hand_type_to_lookup_index(hand_type: PokerHandType) -> int:
        """
        Takes a hand type and returns the index for where you should look
        for its win probability in the table

        This assumes that the PokerHandType is sorted such that
        the first c1 < c2
        """
        if hand_type.c1_rank == hand_type.c2_rank:
            # They must be at the 'beginning of the list'
            return hand_type.c1_rank - 2
        else:
            # 'regular'
            start_offset = 12
            c1_offset = sum(range(12, 14 - hand_type.c1_rank, -1)) * 2
            c2_offset = (hand_type.c2_rank - (hand_type.c1_rank + 1)) * 2
            suit_offset = hand_type.suited
            return 13 + c1_offset + c2_offset + suit_offset

    @staticmethod
    def get_lookup_table():
        return np.load("lookup_tables/preflop.npy")

    @staticmethod
    def gen_hand_types():
        """
        Generate a precomputed list of different hand_types that we can use
        """
        res = []

        # Add in the equal ranks at the beginning
        for rank in RANK_NUM_DICT.values():
            res.append(PokerHandType(rank, rank, False))
        # Loop over hands of different ranks
        for rank in RANK_NUM_DICT.values():
            for rank2 in list(RANK_NUM_DICT.values())[rank - 1 :]:
                for suited in [False, True]:
                    res.append(PokerHandType(rank, rank2, suited))

        pickle.dump(res, open("lookup_tables/hand_types.pkl", "wb+"))

    @staticmethod
    def hand_types() -> List[PokerHandType]:
        """
        Loads the precomputed hand_types
        """
        res = pickle.load(open("lookup_tables/hand_types.pkl", "rb"))

        return res

    @staticmethod
    def generate_hand_win_probabilities(
        nbr_iterations: int = 1000,
        dest: str = "lookup_tables/preflop",
    ) -> np.ndarray:
        result = np.zeros((169, 5))
        for i, hand_type in enumerate(tqdm(PokerOracle.hand_types())):
            c1 = Card("C", NUM_RANK_DICT[hand_type.c1_rank])
            suit2 = "C" if hand_type.suited else "D"
            c2 = Card(suit2, NUM_RANK_DICT[hand_type.c2_rank])

            for num_players in tqdm(range(2, 7), leave=False, desc="# Players"):
                result[
                    i, num_players - 2
                ] = PokerOracle.hole_hand_winning_probability_rollout(
                    [c1, c2], [], num_players=num_players, num_rollouts=nbr_iterations
                )
        if dest != "":
            np.save(dest, result)
        return result

    @staticmethod
    def hole_hand_winning_probability_rollout(
        hole_cards: List[Card],
        public_cards: List[Card],
        num_rollouts: int = 1000,
        num_players: int = 2,
    ) -> float:
        """
        Calculates the probability that the given hole cards will win
        against a random hand of the same size
        """

        if len(hole_cards) != 2 or len(public_cards) > 5:
            raise ValueError(
                "Hole cards must be 2 cards and public cards must be 0-5 cards"
            )

        win_rate = 0.0

        # Initialize a deck without the hole cards
        deck = Deck()
        deck.remove_cards(hole_cards)
        # Also remove any public cards
        deck.remove_cards(public_cards)

        for _ in tqdm(range(num_rollouts), leave=False, desc="Rollouts"):
            d = deck.copy()
            missing_public = 5 - len(public_cards)
            p = public_cards + d.draw(missing_public)
            oponent_cards = [d.draw(2) for _ in range(num_players - 1)]

            hand1 = PokerOracle.evaluate_hand(hole_cards + p)
            oponent_hands = [
                PokerOracle.evaluate_hand(oponent_card + p)
                for oponent_card in oponent_cards
            ]

            won = True
            for oponent in oponent_hands:
                # The evaluation returns a rank -> 1 is better than 10
                if oponent <= hand1:
                    won = False
                    break
            win_rate += won

        win_rate /= num_rollouts

        return win_rate

    @staticmethod
    def get_win_rates_for_hands(
        hands: List[List[Card]], public_cards: List[Card], max_iter: int = 1000
    ) -> np.ndarray:
        """
        Given the n hands and 3, 4 or 5 public cards, returns the win_rate for each player

        This is based on rollouts of the remaining public cards
        """
        if len(public_cards) not in [3, 4, 5]:
            raise ValueError("Must have 3, 4 or 5 public cards")

        for hand in hands:
            for c in hand:
                if c in public_cards:
                    # Cannot allow to have same hand as public cards
                    return np.zeros(len(hands))

        deck = Deck()
        deck.remove_cards(public_cards)
        for hand in hands:
            deck.remove_cards(hand)
        remaining_public = 5 - len(public_cards)

        win_rates = np.zeros(len(hands))
        for _ in range(max_iter):
            d = deck.copy()
            p = public_cards + d.draw(remaining_public)
            hand_values = [PokerOracle.evaluate_hand(hand + p) for hand in hands]
            winner_index = np.argmin(hand_values)
            win_rates[winner_index] += 1
        return win_rates

    @staticmethod
    def get_winner(hands: List[List[Card]], public_cards: List[Card]) -> np.intp:
        """
        Returns the index of the winner of the given hands
        """
        hand_ranks = [PokerOracle.evaluate_hand(hand + public_cards) for hand in hands]

        # lower rank => better hand
        return np.argmin(hand_ranks)

    @staticmethod
    def evaluate_hand(hand: List[Card]) -> int:
        """
        External wrapper to be used for evaluating a hand of cards.
        Simplifies the conversion from the Card class to integers that is needed
        for the hashing
        """

        if len(hand) == 2:
            return PokerOracle._evaluate_hand_without_public_cards(hand)

        int_cards = [c.id for c in hand]
        return PokerOracle._evaluate_hand(int_cards)

    @staticmethod
    def _evaluate_hand_without_public_cards(hand: List[Card]) -> int:
        """
        Used for evaluating utility matrix in the case of no public cards

        Returns a ranking, so lower number is better
        """
        if hand[0].rank == hand[1].rank:
            # pair
            return 14 - hand[0].rank_value
        else:
            return 30 - (np.max([hand[0].rank_value, hand[1].rank_value]))

    @staticmethod
    def _evaluate_hand(cards: List[int]) -> int:
        """
        This is where the magic happens!
        Internal funciton that uses the integer representation of the cards to
        find the precalculated value of a hand of cards.

        this is adapted from https://github.com/HenryRLee/PokerHandEvaluator
        which precomputes tables for evaluating hands of sizes 5-9

        Returns a ranking, so lower number is better
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
