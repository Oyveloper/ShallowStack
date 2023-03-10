from enum import Enum
from typing import List
import numpy as np
from shallowstack.game.action import ActionType, Action
import copy

from shallowstack.player.player import Player
from shallowstack.poker.card import Card, Deck


class PokerGameState(Enum):
    PRE_FLOP = 1
    FLOP = 2
    TURN = 3
    RIVER = 4
    SHOWDOWN = 5


class GameState:
    def __init__(
        self,
        stage: PokerGameState,
        players: List[Player],
        current_player_index: int,
        player_bets: np.ndarray,
        player_checks: np.ndarray,
        players_in_game: np.ndarray,
        pot: int,
        bet_to_match: int,
        public_cards: List[Card],
        deck: Deck,
    ):
        self.deck = deck
        self.stage = stage
        self.players = players
        self.player_bets = player_bets
        self.player_checks = player_checks
        self.players_in_game = players_in_game
        self.current_player_index = current_player_index
        self.bet_to_match = bet_to_match
        self.pot = pot
        self.public_cards = public_cards

    def copy(self):
        return copy.deepcopy(self)


class GameManager:
    def __init__(self, players: List[Player], blind_amount: int = 5):
        """
        Sets up a new game of poker with a given list of players
        The number of players must be between 2 and 6
        """
        if len(players) < 2:
            raise ValueError("A poker game must have at least 2 players")
        if len(players) > 6:
            raise ValueError("This engine only supports up to 6 players")

        self.players = players

        self.blind_amount = blind_amount

        self.small_blind = 1
        self.big_blind = 2 % len(players)

        self.start_game()

    @staticmethod
    def apply_action(state: GameState, action: Action) -> GameState:
        s = state.copy()
        if action.action_type == ActionType.FOLD:
            s.players_in_game[s.current_player_index] = False
            s.player_checks[s.current_player_index] = False
        elif action.action_type == ActionType.CALL:
            diff = s.bet_to_match - s.player_bets[s.current_player_index]
            player = s.players[s.current_player_index]
            if player.can_afford_bet(diff):
                s.player_bets[s.current_player_index] += diff
                player.bet_chips(diff)
                s.pot += diff
            player.bet_chips(diff)
        elif action.action_type == ActionType.RAISE:
            amount = action.amount
            player = s.players[s.current_player_index]
            if player.can_afford_bet(amount):
                s.player_bets[s.current_player_index] += amount
                player.bet_chips(amount)
                s.pot += amount
                s.bet_to_match = s.player_bets[s.current_player_index]
        elif action.action_type == ActionType.CHECK:
            s.player_checks[s.current_player_index] = True
        elif action.action_type == ActionType.ALL_IN:
            player = s.players[s.current_player_index]
            s.player_bets[s.current_player_index] += player.chips
            s.pot += player.chips
            player.bet_chips(player.chips)
            s.player_checks[s.current_player_index] = False
        return s

    @staticmethod
    def get_legal_actions(state: GameState) -> List[ActionType]:
        """
        Returns a list of legal actions for the current player
        """
        actions = [ActionType.FOLD]
        bet_to_match = np.max(state.player_bets)
        player_bet = state.player_bets[state.current_player_index]

        can_afford = state.players[state.current_player_index].can_afford_bet(
            bet_to_match - player_bet
        )
        if player_bet == bet_to_match:
            actions.append(ActionType.CHECK)

        elif can_afford:
            actions += [ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN]

        return actions

    def start_game(self):
        """
        This is where we start a new game of poker
        """
        print()
        print("Starting a new round!")
        # reset game state
        self.game_state = GameState(
            PokerGameState.PRE_FLOP,
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

        # initiate the game by claiming the blinds
        self.claim_blinds()

        # deal cards
        self.deal_cards()

        current_player = (self.big_blind + 1) % len(self.players)
        while True:
            print()

            # Checks to see how game should progress
            if not self.game_state.players_in_game[current_player]:
                # Skip folded players
                continue
            # If all players have checked, we can move on to the next stage
            if np.all(self.game_state.player_checks == self.game_state.players_in_game):
                self.progress_stage()

            if self.game_state.stage == PokerGameState.SHOWDOWN:
                break

            if self.game_state.players_in_game.sum() == 1:
                # Only one player left in the game
                # we have a winner
                break

            self.display()

            # Player needs to do action
            player = self.players[current_player]

            # Find the legal actions
            legal_action_types = GameManager.get_legal_actions(self.game_state)

            # Select acition and execute
            action = player.get_action(legal_action_types)
            self.game_state = GameManager.apply_action(self.game_state, action)

            # move on to next player
            current_player = (current_player + 1) % len(self.players)

            # Display action info
            print(f"Player: {player.name} did: {action.action_type}")
            if action.action_type == ActionType.RAISE:
                print(f"Amount: {action.amount}")

        # Game is over
        # Announce winner
        winner = self.players[self.game_state.players_in_game.argmax()]
        print(f"{winner.name} won the game!")
        print(f"Winnings: {self.game_state.pot}")
        winner.add_chips(self.game_state.pot)
        self.rotate_blinds()

        # Start a new round
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
        self.game_state.players[self.big_blind].bet_chips(self.blind_amount * 2)
        self.game_state.pot += self.blind_amount * 3

    def rotate_blinds(self):
        """
        Rotates the blinds to the next player
        """
        self.small_blind = (self.small_blind + 1) % len(self.players)
        self.big_blind = (self.big_blind + 1) % len(self.players)

    def progress_stage(self):
        if self.game_state.stage == PokerGameState.PRE_FLOP:
            print("Moving on to FLOP. Dealing cards...")
            self.game_state.stage = PokerGameState.FLOP
            self.game_state.public_cards = self.game_state.deck.draw(3)
        elif self.game_state.stage == PokerGameState.FLOP:
            print("Moving on to TURN. Dealing cards...")
            self.game_state.stage = PokerGameState.TURN
            self.game_state.public_cards += self.game_state.deck.draw(1)
        elif self.game_state.stage == PokerGameState.TURN:
            print("Moving on to RIVER. Dealing cards...")
            self.game_state.stage = PokerGameState.RIVER
            self.game_state.public_cards += self.game_state.deck.draw(1)
        elif self.game_state.stage == PokerGameState.RIVER:
            self.game_state.stage = PokerGameState.SHOWDOWN
            print("SHOWDOWN")

        self.player_checks = np.zeros(len(self.players), dtype=bool)

    def deal_cards(self):
        for player in self.game_state.players:
            player.receive_cards(self.game_state.deck.draw(2))

    def showdown(self):
        pass

    def display(self):
        print(f"Pot: { self.game_state.pot}")
        print(f"Public cards: {self.game_state.public_cards}")
        for i, player in enumerate(self.game_state.players):
            print(f"{player.name}: {player.hand} ({self.game_state.player_bets[i]})")

        print()
