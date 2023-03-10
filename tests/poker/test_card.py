import numpy as np
from numpy.lib import math

from shallowstack.poker.card import (
    Card,
    Deck,
    hole_card_ids_from_pair_idx,
    hole_pair_idx_from_ids,
)


def test_hole_pair_idx_from_ids():
    hole_pair_idxes = np.array([])

    deck = Deck()

    for i in range(len(deck.cards)):
        card1 = deck.cards[i]
        id1 = card1.id
        for j in range(i + 1, len(deck.cards)):
            card2 = deck.cards[j]
            id2 = card2.id
            hole_pair_id = hole_pair_idx_from_ids(id1, id2)
            print(hole_pair_id)
            hole_pair_idxes = np.append(hole_pair_idxes, hole_pair_id)

    assert len(hole_pair_idxes) == 1326
    assert np.min(hole_pair_idxes) == 0
    assert np.max(hole_pair_idxes) == 1325


def test_hole_cards_from_pair_idx():
    deck = Deck()

    for i in range(len(deck.cards)):
        card1 = deck.cards[i]
        id1 = card1.id
        for j in range(i + 1, len(deck.cards)):
            card2 = deck.cards[j]
            id2 = card2.id
            hole_pair_idx = hole_pair_idx_from_ids(id1, id2)
            hole_cards = hole_card_ids_from_pair_idx(hole_pair_idx)
            assert hole_cards[0] == id1
            assert hole_cards[1] == id2


def test_card_in_list_check():
    c1 = Card("H", "J")
    c2 = Card("H", "J")
    c3 = Card("S", "J")

    l = [c1, c3]

    assert c1 == c2
    assert c1 in l
    assert c2 in l


def test_deck_order():
    deck = Deck()

    res = []
    for card in deck.cards:
        res.append(card.id)
        print(card.id)

    assert res == list(range(52))


def test_remove_card():
    deck = Deck()
    cards = deck.cards.copy()

    # ignore last card as this destroys the distribution
    for card in cards[:-1]:
        deck.remove_cards([card])
        assert deck.card_distribution[card.id] == 0
        assert math.isclose(np.sum(deck.card_distribution), 1.0)
