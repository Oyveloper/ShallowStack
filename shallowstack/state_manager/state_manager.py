import copy
from enum import Enum
from typing import List, Optional, Tuple
from shallowstack.game.action import ALLOWED_RAISES, Action, ActionType
from shallowstack.player.player import Player
from shallowstack.poker.card import Card, Deck
import numpy as np

from shallowstack.poker.poker_oracle import PokerOracle

BET_PER_STAGE_LIMIT = 2


class PokerGameStage(Enum):
    PRE_FLOP = 1
    FLOP = 2
    TURN = 3
    RIVER = 4
    SHOWDOWN = 5


class PokerGameStateType(Enum):
    PLAYER = 0
    DEALER = 1
    WINNER = 2


class GameState:
    def __init__(
        self,
        stage: PokerGameStage,
        players: List[Player],
        current_player_index: int,
        player_bets: np.ndarray,
        player_checks: np.ndarray,
        players_in_game: np.ndarray,
        pot: int,
        bet_to_match: int,
        public_cards: List[Card],
        deck: Deck,
        game_state_type: PokerGameStateType = PokerGameStateType.PLAYER,
        winner: Optional[Player] = None,
        winner_index: int = -1,
        stage_bet_count: int = 0,
        last_action: Optional[Action] = None,
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
        self.game_state_type = game_state_type
        self.winner: Optional[Player] = winner
        self.winner_index: int = winner_index
        self.stage_bet_count = stage_bet_count
        self.last_action = last_action

    def copy(self):
        return copy.deepcopy(self)

    def increment_player_index(self):
        self.current_player_index = (self.current_player_index + 1) % len(self.players)

    def reset_for_new_round(self):
        # reset game state
        self.pot = 0
        self.bet_to_match = 0
        self.player_bets = np.zeros(len(self.players))
        self.player_checks = np.zeros(len(self.players))
        self.players_in_game = np.ones(len(self.players))
        self.deck = Deck()
        self.stage = PokerGameStage.PRE_FLOP
        self.public_cards = []
        self.game_state_type = PokerGameStateType.PLAYER
        self.winner = None
        self.stage_bet_count = 0
        for player in self.players:
            player.prepare_for_new_round()


class StateManager:
    @staticmethod
    def get_child_states(
        state: GameState, nbr_random_events: int
    ) -> List[Tuple[Action, GameState]]:
        """
        Generates child states for a given state
        """
        states = []
        if state.game_state_type == PokerGameStateType.PLAYER:
            states = StateManager.get_actions_with_new_states(state)
        elif state.game_state_type == PokerGameStateType.DEALER:
            for _ in range(nbr_random_events):
                new_state = state.copy()
                deck = Deck()
                deck.remove_cards(new_state.public_cards)
                states.append((None, StateManager.progress_stage(new_state, deck)))

        return states

    @staticmethod
    def get_actions_with_new_states(
        state: GameState,
    ) -> List[Tuple[Action, GameState]]:
        game_state = state.copy()
        action_types = StateManager.get_legal_actions(game_state)

        result = []
        for action_type in action_types:
            # Generalize the generation so that we can have a fixed
            # amount of allowed raises exposed to re-solvers
            amounts = [0]
            if action_type == ActionType.RAISE:
                amounts = ALLOWED_RAISES
            elif action_type == ActionType.ALL_IN:
                amounts = [game_state.players[game_state.current_player_index].chips]

            for amount in amounts:
                if game_state.players[game_state.current_player_index].can_afford_bet(
                    amount
                ):
                    action = Action(action_type, amount)
                    new_state = StateManager.apply_action(game_state, action)

                    result.append((action, new_state))

        return result

    @staticmethod
    def apply_action(state: GameState, action: Action) -> GameState:
        s = state.copy()
        pot_raised = False

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
                s.player_checks[s.current_player_index] = True
        elif action.action_type == ActionType.CHECK:
            s.player_checks[s.current_player_index] = True

        elif action.action_type == ActionType.RAISE:
            pot_raised = True
            amount = action.amount
            player = s.players[s.current_player_index]

            diff = max(0, s.bet_to_match - s.player_bets[s.current_player_index])
            total = diff + amount
            if player.can_afford_bet(total):
                s.player_bets[s.current_player_index] += total
                player.bet_chips(total)
                s.pot += total
                s.bet_to_match = s.player_bets[s.current_player_index]

        elif action.action_type == ActionType.ALL_IN:
            player = s.players[s.current_player_index]
            s.player_bets[s.current_player_index] += player.chips
            s.pot += player.chips
            s.bet_to_match = s.player_bets[s.current_player_index]
            player.bet_chips(player.chips)
            s.player_checks[s.current_player_index] = False
            pot_raised = True

            player.went_all_in = True

        if pot_raised:
            s.player_checks = np.zeros(len(s.players), dtype=bool)
            s.player_checks[s.current_player_index] = True
            s.stage_bet_count += 1
        if np.all(s.player_checks == s.players_in_game):
            s.game_state_type = PokerGameStateType.DEALER
        if np.sum(s.players_in_game) == 1:
            s.game_state_type = PokerGameStateType.WINNER
            s.winner_index = int(np.argmax(s.players_in_game))
            s.winner = s.players[s.winner_index]

        s.last_action = action
        s.increment_player_index()
        return s

    @staticmethod
    def get_legal_actions(state: GameState) -> List[ActionType]:
        """
        Returns a list of legal actions for the current player
        """
        actions = [ActionType.FOLD]
        bet_to_match = np.max(state.player_bets)
        player_bet = state.player_bets[state.current_player_index]

        diff = max(0, bet_to_match - player_bet)
        player = state.players[state.current_player_index]
        can_afford_call = player.can_afford_bet(diff)

        if diff == 0 or player.went_all_in:
            actions.append(ActionType.CHECK)

        if can_afford_call and not state.player_checks[state.current_player_index]:
            actions.append(ActionType.CALL)

        # Check if they can afford the bare minimum raise
        can_afford_raise = player.can_afford_bet(max(0, diff) + 1)

        if can_afford_raise and state.stage_bet_count < BET_PER_STAGE_LIMIT:
            actions.append(ActionType.RAISE)

        if player.chips > 0 and state.stage_bet_count < BET_PER_STAGE_LIMIT:
            actions.append(ActionType.ALL_IN)

        return actions

    @staticmethod
    def find_winner(state: GameState) -> GameState:
        """
        finds the winner of the current state, and returns modified gamestate
        """
        s = state.copy()
        remaining_players: List[Player] = [
            s.players[i] for i in range(len(s.players)) if s.players_in_game[i]
        ]
        hands = [p.hand for p in remaining_players]
        winner_index = PokerOracle.get_winner(hands, s.public_cards)
        s.winner = remaining_players[winner_index]
        s.winner_index = winner_index
        s.game_state_type = PokerGameStateType.WINNER

        return s

    @staticmethod
    def progress_stage(state: GameState, deck: Deck) -> GameState:
        """
        Creates a new state from a stage transition
        """
        s = state.copy()
        s.player_checks = np.zeros(len(s.players), dtype=bool)
        s.player_checks[s.current_player_index] = True
        s.game_state_type = PokerGameStateType.PLAYER
        s.stage_bet_count = 0
        if s.stage == PokerGameStage.PRE_FLOP:
            s.stage = PokerGameStage.FLOP
            s.public_cards = deck.draw(3)
        elif s.stage == PokerGameStage.FLOP:
            s.stage = PokerGameStage.TURN
            s.public_cards += deck.draw(1)
        elif s.stage == PokerGameStage.TURN:
            s.stage = PokerGameStage.RIVER
            s.public_cards += deck.draw(1)
        elif s.stage == PokerGameStage.RIVER:
            s.stage = PokerGameStage.SHOWDOWN

        return s
