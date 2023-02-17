from shallowstack.poker.card import Deck
from shallowstack.poker.poker_oracle import PokerOracle


def main():
    deck = Deck()

    hand = deck.draw(6)
    print("Hand: ", hand)

    evaluated_cards = PokerOracle.evaluate_hand(hand)
    print("Evaluated cards: ", evaluated_cards)


if __name__ == "__main__":
    main()
