from shallowstack.poker.card import Card, hole_pair_idx_from_ids
from shallowstack.poker.poker_oracle import PokerOracle


def test_poker_orakle_generate_hand_types():
    assert len(PokerOracle.hand_types()) == 169


def test_poker_orakle_hand_type_index():
    types = PokerOracle.hand_types()

    for t in types:
        print(t)
        print(PokerOracle.hand_type_to_lookup_index(t))
        assert PokerOracle.hand_type_to_lookup_index(t) == types.index(t)


def test_utility_matrix():
    public_cards = [
        Card("H", "J"),
        Card("S", "8"),
        Card("H", "4"),
    ]
    m = PokerOracle.calculate_utility_matrix(public_cards)

    # Make sure dimmensions are correct
    assert m.shape == (1326, 1326)

    # Make sure diagonal is correct
    for i in range(1326):
        assert m[i, i] == 0

    # Prechecked value
    hand1 = [Card("H", "10"), Card("S", "10")]
    hand2 = [Card("S", "Q"), Card("S", "9")]

    h1_idx = hole_pair_idx_from_ids(hand1[0].id, hand1[1].id)
    h2_idx = hole_pair_idx_from_ids(hand2[0].id, hand2[1].id)
    assert m[h1_idx, h2_idx] == 1
    assert m[h2_idx, h1_idx] == -1
    assert m[h1_idx, h1_idx] == 0
