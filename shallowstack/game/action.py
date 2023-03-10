from enum import Enum


class ActionType(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE = 3
    ALL_IN = 4


class Action:
    def __init__(self, action_type: ActionType, amount: int = 0):
        self.action_type: ActionType = action_type
        self.amount: int = amount
