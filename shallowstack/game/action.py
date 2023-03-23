from enum import Enum


class ActionType(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    ALL_IN = 3
    RAISE = 4


class Action:
    def __init__(self, action_type: ActionType, amount: int = 0):
        self.action_type: ActionType = action_type
        self.amount: int = amount


ALLOWED_RAISES = [
    5,
    10,
]

AGENT_ACTIONS = [
    Action(ActionType.FOLD),
    Action(ActionType.CHECK),
    Action(ActionType.CALL),
    Action(ActionType.ALL_IN),
    Action(ActionType.RAISE, 5),
    Action(ActionType.RAISE, 10),
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
