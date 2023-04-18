import copy
from enum import Enum
from typing import List, Tuple
from shallowstack.config.config import POKER_CONFIG
from shallowstack.game.action import ALLOWED_RAISES, Action, ActionType
from shallowstack.poker.card import Card, Deck
import numpy as np


BET_PER_STAGE_LIMIT = POKER_CONFIG.getint("BET_PER_STAGE_LIMIT")


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
        current_player_index: int,
        player_bets: np.ndarray,
        player_chips: np.ndarray,
        player_checks: np.ndarray,
        players_in_game: np.ndarray,
        players_all_in: np.ndarray,
        pot: int,
        bet_to_match: int,
        public_info: List[Card],
        deck: Deck,
        game_state_type: PokerGameStateType = PokerGameStateType.PLAYER,
        winner_index: int = -1,
        stage_bet_count: int = 0,
    ):
        self.deck = deck
        self.stage = stage
        self.player_bets = player_bets
        self.player_chips = player_chips
        self.player_checks = player_checks
        self.players_in_game = players_in_game
        self.players_all_in = players_all_in
        self.current_player_index = current_player_index
        self.bet_to_match = bet_to_match
        self.pot = pot
        self.public_info = public_info
        self.game_state_type = game_state_type
        self.winner_index: int = winner_index
        self.stage_bet_count = stage_bet_count

    def copy(self):
        return copy.deepcopy(self)

    def increment_player_index(self):
        self.current_player_index = (self.current_player_index + 1) % len(
            self.player_bets
        )

    def reset_for_new_round(self, redistribute_chips: bool = False):
        # reset game state
        self.pot = 0
        self.bet_to_match = 0
        self.player_bets = np.zeros(len(self.player_bets))
        self.player_checks = np.zeros(len(self.player_bets))
        self.players_in_game = np.ones(len(self.player_bets))
        self.players_all_in = np.zeros(len(self.player_bets))
        self.deck = Deck()
        self.stage = PokerGameStage.PRE_FLOP
        self.public_info = []
        self.game_state_type = PokerGameStateType.PLAYER
        self.stage_bet_count = 0

        if redistribute_chips:
            self.player_chips = np.ones(len(self.player_bets)) * 1000


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
                deck.remove_cards(new_state.public_info)
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
                amounts = [game_state.player_chips[game_state.current_player_index]]

            for amount in amounts:
                if StateManager.can_afford_bet(
                    game_state.current_player_index, amount, game_state
                ):
                    action = Action(action_type, amount)
                    new_state = StateManager.apply_action(game_state, action)

                    result.append((action, new_state))

        return result

    @staticmethod
    def can_afford_bet(player_index: int, amount: float, state: GameState) -> bool:
        """Checks if given player can afford bet"""
        return state.player_chips[player_index] >= amount

    @staticmethod
    def bet_amount(player_index: int, amount, state: GameState) -> GameState:
        """Bets given amount for given player"""
        s = state.copy()
        s.player_chips[player_index] -= amount
        s.player_bets[player_index] += amount
        s.pot += amount

        if s.player_bets[player_index] > s.bet_to_match:
            s.bet_to_match = s.player_bets[player_index]

        return s

    @staticmethod
    def apply_action(state: GameState, action: Action) -> GameState:
        s = state.copy()
        pot_raised = False

        if action.action_type == ActionType.FOLD:
            s.players_in_game[s.current_player_index] = False
            s.player_checks[s.current_player_index] = False

        elif action.action_type == ActionType.CALL:
            diff = s.bet_to_match - s.player_bets[s.current_player_index]

            if StateManager.can_afford_bet(s.current_player_index, diff, s):
                s = StateManager.bet_amount(s.current_player_index, diff, s)
                s.player_checks[s.current_player_index] = True

        elif action.action_type == ActionType.CHECK:
            s.player_checks[s.current_player_index] = True

        elif action.action_type == ActionType.RAISE:
            pot_raised = True
            amount = action.amount

            diff = max(0, s.bet_to_match - s.player_bets[s.current_player_index])
            total = diff + amount
            if StateManager.can_afford_bet(s.current_player_index, total, s):
                s = StateManager.bet_amount(s.current_player_index, total, s)

        elif action.action_type == ActionType.ALL_IN:
            amount = s.player_chips[s.current_player_index]
            s = StateManager.bet_amount(s.current_player_index, amount, s)
            s.player_checks[s.current_player_index] = False
            pot_raised = True

            s.players_all_in[s.current_player_index] = True

        if pot_raised:
            s.player_checks = np.zeros(len(s.player_bets), dtype=bool)
            s.player_checks[s.current_player_index] = True
            s.stage_bet_count += 1
        if np.all(s.player_checks == s.players_in_game):
            s.game_state_type = PokerGameStateType.DEALER
        if np.sum(s.players_in_game) == 1:
            s.game_state_type = PokerGameStateType.WINNER
            s.winner_index = int(np.argmax(s.players_in_game))

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
        can_afford_call = StateManager.can_afford_bet(
            state.current_player_index, diff, state
        )
        all_in = state.players_all_in[state.current_player_index]

        if diff == 0 or all_in:
            actions.append(ActionType.CHECK)

        if can_afford_call and not state.player_checks[state.current_player_index]:
            actions.append(ActionType.CALL)

        # Check if they can afford the bare minimum raise
        can_afford_raise = StateManager.can_afford_bet(
            state.current_player_index, max(0, diff) + 1, state
        )

        if can_afford_raise and state.stage_bet_count < BET_PER_STAGE_LIMIT:
            actions.append(ActionType.RAISE)

        if (
            state.player_chips[state.current_player_index] > 0
            and state.stage_bet_count < BET_PER_STAGE_LIMIT
        ):
            actions.append(ActionType.ALL_IN)

        return actions

    @staticmethod
    def progress_stage(state: GameState, deck: Deck) -> GameState:
        """
        Creates a new state from a stage transition
        """
        s = state.copy()
        s.player_checks = np.zeros(len(s.player_bets), dtype=bool)
        s.player_checks[s.current_player_index] = True
        s.game_state_type = PokerGameStateType.PLAYER
        s.stage_bet_count = 0
        if s.stage == PokerGameStage.PRE_FLOP:
            s.stage = PokerGameStage.FLOP
            s.public_info = deck.draw(3)
        elif s.stage == PokerGameStage.FLOP:
            s.stage = PokerGameStage.TURN
            s.public_info += deck.draw(1)
        elif s.stage == PokerGameStage.TURN:
            s.stage = PokerGameStage.RIVER
            s.public_info += deck.draw(1)
        elif s.stage == PokerGameStage.RIVER:
            s.stage = PokerGameStage.SHOWDOWN

        s.deck = deck

        return s
