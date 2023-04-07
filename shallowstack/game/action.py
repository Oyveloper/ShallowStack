from enum import Enum

from shallowstack.config.config import POKER_CONFIG


class ActionType(Enum):
    FOLD = 0
    CALL = 1
    CHECK = 2
    ALL_IN = 3
    RAISE = 4


class Action:
    def __init__(self, action_type: ActionType, amount: int = 0):
        self.action_type: ActionType = action_type
        self.amount: int = amount

    def __eq__(self, other):
        return self.action_type == other.action_type and self.amount == other.amount


ALLOWED_RAISES = [
    POKER_CONFIG.getint("SMALL_BLIND"),
    POKER_CONFIG.getint("BIG_BLIND"),
]

AGENT_ACTIONS = [
    Action(ActionType.FOLD),
    Action(ActionType.CALL),
    Action(ActionType.CHECK),
    Action(ActionType.ALL_IN),
    Action(ActionType.RAISE, ALLOWED_RAISES[0]),
    Action(ActionType.RAISE, ALLOWED_RAISES[1]),
]


def agent_action_index(action: Action) -> int:
    """
    Calculates the index in the array over actions allowed for
    artificial agents
    """
    res = action.action_type.value
    if action.action_type == ActionType.RAISE:
        if not action.amount in ALLOWED_RAISES:
            raise ValueError("Invalid raise amount")
        res += ALLOWED_RAISES.index(action.amount)
    return res
