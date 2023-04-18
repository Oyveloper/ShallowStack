from typing import List, Optional
import numpy as np
from shallowstack.config.config import POKER_CONFIG
from shallowstack.game.action import ActionType

from shallowstack.player import Player, Human, ResolvePlayer, RolloutPlayer
from shallowstack.player.hybrid_player import HybridPlayer
from shallowstack.poker.card import Deck
from shallowstack.poker.poker_oracle import PokerOracle
from shallowstack.state_manager.state_manager import (
    GameState,
    PokerGameStage,
    PokerGameStateType,
    StateManager,
)


PLAYER_CONFIGS = {
    "HH": [
        Human("Human 1"),
        Human("Human2 2"),
    ],
    "HRes": [Human("Human"), ResolvePlayer("ResolvePlayer")],
    "ResRes": [ResolvePlayer("Resolver 1"), ResolvePlayer("Resolver 2")],
    "ResRoll": [ResolvePlayer("Resolver 1"), RolloutPlayer("Rollout 2")],
    "Custom": [],
    "HRoll": [Human("Human"), RolloutPlayer("Rollout")],
}


def custom_player_setup() -> List[Player]:
    players = []
    nbr_players = 0
    while nbr_players < 2:
        try:
            nbr_players = int(input("How many players? "))
        except KeyboardInterrupt:
            exit()
        except:
            print("Must be between 2 and 6")

    options = ["Human", "Rollout"]
    if nbr_players == 2:
        options.append("Resolve")
        options.append("Hybrid")
    for i in range(nbr_players):
        print(f"Setup player {i}")
        ok = False
        while not ok:
            ok = True
            player_type = input(f"Player {i + 1} type? ({'/'.join(options)})\n").strip()
            if player_type == "Human":
                players.append(Human(f"Human {i + 1}"))
            elif player_type == "Resolve":
                if nbr_players != 2:
                    print("Cannot use resolve with more than 2 players")
                    ok = False
                    continue
                players.append(ResolvePlayer(f"Resolve {i + 1}"))
            elif player_type == "Rollout":
                players.append(RolloutPlayer(f"Rollout {i + 1}"))
            elif player_type == "Hybrid":
                if nbr_players != 2:
                    print("Cannot use resolve with more than 2 players")
                    ok = False
                    continue

                prob_ok = False
                prob: float = 0.0
                while not prob_ok:
                    try:
                        prob = float(input("Probability? "))
                        prob_ok = True
                    except KeyboardInterrupt:
                        exit()
                    except:
                        print("Must be a number between 0 and 1")
                players.append(
                    HybridPlayer(f"Hybrid {i + 1}", resolve_probability=prob)
                )
            else:
                ok = False
                print("Invalid player type")

    return players


class GameManager:
    def __init__(self, players: List[Player], show_private_info: bool = False):
        """
        Sets up a new game of poker with a given list of players
        The number of players must be between 2 and 6
        """
        if len(players) < 2:
            raise ValueError("A poker game must have at least 2 players")
        if len(players) > 6:
            raise ValueError("This engine only supports up to 6 players")

        self.players = players

        self.blind_amount = POKER_CONFIG.getint("SMALL_BLIND")

        self.small_blind = 1
        self.big_blind = 2 % len(players)

        self.game_state = GameState(
            PokerGameStage.PRE_FLOP,
            0,
            np.zeros(len(self.players)),
            np.ones(len(self.players)) * 1000,
            np.zeros(len(self.players)),
            np.ones(len(self.players)),
            np.zeros(len(self.players), dtype=bool),
            0,
            0,
            [],
            Deck(),
        )

        self.winner: Optional[Player] = None
        self.show_private_info = show_private_info

        self.game_stats = [{"player": player.name, "wins": 0} for player in players]

    def start_game(self, nbr_rounds: int = 10):
        """
        This is where we start a new game of poker
        """
        print(f"\n{'-' * 20}")
        print("Starting a new round!")
        # REset for new game
        self.game_state.reset_for_new_round(
            POKER_CONFIG.getboolean("REDISTRIBUTE_CHIPS")
        )

        for player in self.players:
            player.prepare_for_new_round()

        # initiate the game by claiming the blinds
        self.claim_blinds()

        # deal cards
        self.deal_cards()

        while True:
            # move on to next player

            self.display()
            if self.game_state.game_state_type == PokerGameStateType.WINNER:
                self.winner = self.players[self.game_state.winner_index]
                break

            elif self.game_state.game_state_type == PokerGameStateType.DEALER:
                print("Dealing new cards")
                self.game_state = StateManager.progress_stage(
                    self.game_state, self.game_state.deck
                )
                continue

            if self.game_state.stage == PokerGameStage.SHOWDOWN:
                self.showdown()
                break

            print()

            # Checks to see how game should progress
            if not self.game_state.players_in_game[
                self.game_state.current_player_index
            ]:
                # Skip folded players
                continue

            # Player needs to do action
            player = self.players[self.game_state.current_player_index]

            # Select acition and execute
            action = player.get_action(self.game_state)
            self.game_state = StateManager.apply_action(self.game_state, action)
            for p in self.players:
                p.inform_of_action(action, player)

            # Display action info
            print(f"Player: {player.name} did: {action.action_type}")
            if action.action_type == ActionType.RAISE:
                print(f"Amount: {action.amount}")

        # Game is over
        # Announce winner

        if self.winner is not None:
            print(f"{self.winner.name} won the game!")
            print(f"Winnings: {self.game_state.pot}")

            total_index = self.players.index(self.winner)
            self.game_state.player_chips[total_index] += self.game_state.pot

            self.update_stats(total_index, self.game_state.pot)

        self.rotate_blinds()

        # Start a new round

        for player in self.players:
            if player.chips <= 0:
                print(f"{player.name} is out of chips!")
                return

        if nbr_rounds == 0:
            self.print_stats()
            return

        self.start_game(nbr_rounds - 1)

    def showdown(self):
        """Showdown to find the winner"""
        print("--- SHOWDOWN! ---")
        remaining_players = []
        for i, player in enumerate(self.players):
            if self.game_state.players_in_game[i]:
                remaining_players.append(player)

        hands = [p.hand for p in remaining_players]
        winner_index = PokerOracle.get_winner(hands, self.game_state.public_info)
        self.winner = remaining_players[winner_index]

    def claim_blinds(self):
        """
        Claims the blinds before the game start
        """
        print("Claiming blinds")
        print(f"Small blind: {self.players[self.small_blind].name}")
        print(f"Big blind: {self.players[self.big_blind].name}")
        print()
        self.game_state = StateManager.bet_amount(
            self.small_blind, self.blind_amount, self.game_state
        )
        self.game_state = StateManager.bet_amount(
            self.big_blind, self.blind_amount * 2, self.game_state
        )

        self.game_state.current_player_index = (self.big_blind + 1) % len(self.players)

    def rotate_blinds(self):
        """
        Rotates the blinds to the next player
        """
        self.small_blind = (self.small_blind + 1) % len(self.players)
        self.big_blind = (self.big_blind + 1) % len(self.players)

        self.player_checks = np.zeros(len(self.players), dtype=bool)

    def deal_cards(self):
        for player in self.players:
            player.receive_cards(self.game_state.deck.draw(2))

    def display(self):
        print()
        print(f"Bet to match: {self.game_state.bet_to_match}")
        print(f"Pot: { self.game_state.pot}")
        print(f"Public cards: {self.game_state.public_info}")
        for i, player in enumerate(self.players):
            hand_str = f"{player.hand}" if self.show_private_info else ""
            print(
                f"{player.name}: {hand_str} (bets: {self.game_state.player_bets[i]}), chips: {self.game_state.player_chips[i]}"
            )

        print()

    def update_stats(self, winner_index: int, winnings: float):
        """Updates the stat object"""
        self.game_stats[winner_index]["wins"] += winnings

    def print_stats(self):
        """
        Prints the stats after a game
        """

        print("Game stats:")
        for stat in self.game_stats:
            print(f"{stat['player']}: {stat['wins']}")
