from typing import List
import numpy as np
from shallowstack.config.config import POKER_CONFIG
from shallowstack.game.action import ActionType

from shallowstack.player import Player, Human, ResolvePlayer, RolloutPlayer
from shallowstack.poker.card import Deck
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
    "Custom": [],
    "HRoll": [Human("Human"), RolloutPlayer("Rollout")],
}


class GameManager:
    def __init__(self, players: List[Player]):
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
            self.players,
            0,
            np.zeros(len(self.players)),
            np.zeros(len(self.players)),
            np.ones(len(self.players)),
            0,
            0,
            [],
            Deck(),
        )

    def setup_game(self):
        pass

    def start_game(self):
        """
        This is where we start a new game of poker
        """
        print()
        print("Starting a new round!")
        # REset for new game
        self.game_state.reset_for_new_round()

        # initiate the game by claiming the blinds
        self.claim_blinds()

        # deal cards
        self.deal_cards()

        while True:
            # move on to next player

            self.display()
            if self.game_state.game_state_type == PokerGameStateType.WINNER:
                break
            elif self.game_state.game_state_type == PokerGameStateType.DEALER:
                self.game_state = StateManager.progress_stage(
                    self.game_state, self.game_state.deck
                )
                continue

            if self.game_state.stage == PokerGameStage.SHOWDOWN:
                self.game_state = StateManager.find_winner(self.game_state)
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
            for p in self.game_state.players:
                p.inform_of_action(action, player)

            # Display action info
            print(f"Player: {player.name} did: {action.action_type}")
            if action.action_type == ActionType.RAISE:
                print(f"Amount: {action.amount}")

        # Game is over
        # Announce winner
        winner = self.game_state.winner
        if winner is not None:
            print(f"{winner.name} won the game!")
            print(f"Winnings: {self.game_state.pot}")
            winner.add_chips(self.game_state.pot)
        self.rotate_blinds()

        # Start a new round

        for player in self.players:
            if player.chips <= 0:
                print(f"{player.name} is out of chips!")
                return

        self.start_game()

    def claim_blinds(self):
        """
        Claims the blinds before the game start
        """
        print("Claiming blinds")
        print(f"Small blind: {self.players[self.small_blind].name}")
        print(f"Big blind: {self.players[self.big_blind].name}")
        print()
        self.game_state.players[self.small_blind].bet_chips(self.blind_amount)
        self.game_state.player_bets[self.small_blind] = self.blind_amount

        self.game_state.players[self.big_blind].bet_chips(self.blind_amount * 2)
        self.game_state.player_bets[self.big_blind] = self.blind_amount * 2

        self.game_state.pot += self.blind_amount * 3
        self.game_state.bet_to_match = self.blind_amount * 2
        self.game_state.current_player_index = (self.big_blind + 1) % len(self.players)

    def rotate_blinds(self):
        """
        Rotates the blinds to the next player
        """
        self.small_blind = (self.small_blind + 1) % len(self.players)
        self.big_blind = (self.big_blind + 1) % len(self.players)

        self.player_checks = np.zeros(len(self.players), dtype=bool)

    def deal_cards(self):
        for player in self.game_state.players:
            player.receive_cards(self.game_state.deck.draw(2))

    def showdown(self):
        pass

    def display(self):
        print(f"Bet to match: {self.game_state.bet_to_match}")
        print(f"Pot: { self.game_state.pot}")
        print(f"Public cards: {self.game_state.public_cards}")
        for i, player in enumerate(self.game_state.players):
            print(
                f"{player.name}: {player.hand} (bets: {self.game_state.player_bets[i]}), chips: {self.game_state.players[i].chips}"
            )

        print()
