import sys

from shallowstack.game.poker_game import GameManager
from shallowstack.player.human import Human
from shallowstack.player.resolve_player import ResolvePlayer
from shallowstack.poker.card import Card
from shallowstack.poker.poker_oracle import PokerOracle


def generate_cheat_sheet():
    PokerOracle.generate_hand_win_probabilities(nbr_iterations=1000)


def show_cheat_sheet():
    table = PokerOracle.get_lookup_table()
    print(table)

    c1 = Card("C", "A")
    c2 = Card("S", "A")
    hand_type = PokerOracle.hand_to_hand_type([c1, c2])
    idx = PokerOracle.hand_type_to_lookup_index(hand_type)
    print(f"Probabilities for {c1} {c2} are: ")
    print(table[idx, :])
    print(f"index is {idx}")


def main():
    player1 = Human("Player 1")
    player2 = ResolvePlayer("Player 2")

    game = GameManager([player1, player2])


if __name__ == "__main__":
    args = sys.argv[1:]
    print(args)
    if len(args) > 0:
        a0 = args[0]
        if a0 == "cheat_sheet":
            generate_cheat_sheet()
        elif a0 == "show_cheat_sheet":
            show_cheat_sheet()
        elif a0 == "gen_hand_types":
            PokerOracle.gen_hand_types()
    else:
        main()
