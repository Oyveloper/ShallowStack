from shallowstack.game.action import ActionType, Action
from shallowstack.player.player import Player
from shallowstack.state_manager.state_manager import StateManager, GameState


class Human(Player):
    def __init__(self, name: str, chips: int = 1000):
        super().__init__(name, chips)

    def get_action(self, game_state: GameState) -> Action:
        legal_actions = StateManager.get_legal_actions(game_state)

        print(f"Player: {self.name} must chose an action")
        action_descriptions = [
            f"{action.name} [{i}]" for i, action in enumerate(legal_actions)
        ]

        print(f"Legal Actions: \n{' | '.join(action_descriptions)}")

        has_action = False
        action_index = 0
        while not has_action:
            try:
                action_input = input("Enter Action: ")
                print(action_input)
                action_index = int(action_input)
                assert action_index in range(len(legal_actions))
                has_action = True
            except KeyboardInterrupt:
                exit()
            except:
                print("Invalid")

        action_type = legal_actions[action_index]
        if action_type == ActionType.RAISE:
            raise_amount = 0
            has_amount = False
            while not has_amount:
                try:
                    raise_amount = int(input("Enter Raise Amount: "))
                    assert raise_amount > 0
                    has_amount = True
                except KeyboardInterrupt:
                    exit()
                except:
                    print("Invalid")

            return Action(action_type, raise_amount)

        return Action(action_type)
