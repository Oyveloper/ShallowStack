from shallowstack.game.poker_game import PLAYER_CONFIGS, GameManager
from shallowstack.neural_net.datamodule import PokerDataModule
from shallowstack.neural_net.neural_net_trainer import NNTrainer
from shallowstack.player.human import Human
from shallowstack.player.hybrid_player import HybridPlayer
from shallowstack.player.resolve_player import ResolvePlayer
from shallowstack.player.rollout_player import RolloutPlayer
from shallowstack.poker.card import Card
from shallowstack.poker.poker_oracle import PokerOracle

import debugpy
import click

from shallowstack.state_manager.state_manager import PokerGameStage


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug: bool):
    if debug:
        debugpy.listen(5678)
        debugpy.wait_for_client()


@cli.command()
@click.option(
    "--stage",
    default="RIVER",
    type=click.Choice([el for el in PokerGameStage.__members__]),
)
@click.option(
    "--size",
    default=1000,
    type=click.IntRange(min=1, max=1000000),
)
@click.option(
    "--override/--no-override",
    default=False,
    type=click.BOOL,
)
def generate_training_data_single_stage(stage: str, size: int, override: bool):
    try:
        s = PokerGameStage[stage]
    except:
        print("Invalid stage")
        return

    d = PokerDataModule(s, 1, size, force_override=override)
    d.setup("")


@click.command()
def generate_training_data():
    for stage in PokerGameStage:
        generate_training_data_single_stage(stage.name, 1000, False)


@cli.command()
@click.option(
    "--stage",
    default="RIVER",
    type=click.Choice([el for el in PokerGameStage.__members__]),
)
def train_single_network(stage: str):
    try:
        s = PokerGameStage[stage]
    except:
        print("Invalid stage")
        return

    nn_trainer = NNTrainer()

    nn_trainer.train_network(s, 100)


@cli.command()
@click.option(
    "--epochs",
    default=1000,
    type=click.IntRange(min=1, max=1000000),
)
@click.option(
    "--data_size",
    default=1000,
    type=click.IntRange(min=1, max=1000000),
)
@click.option(
    "--override_river/--no-override_river",
    default=False,
    type=click.BOOL,
)
def train_all(epochs: int, data_size: int, override_river: bool):
    nn_trainer = NNTrainer()
    nn_trainer.train_all_networks(epochs, data_size, override_river)


@cli.command()
def test_stuff():
    data = PokerDataModule(PokerGameStage.RIVER, 1)
    data.setup("")

    i = iter(data.train_dataloader())
    print(next(i))


@cli.command()
def generate_cheat_sheet():
    PokerOracle.generate_hand_win_probabilities(nbr_iterations=1000)


@cli.command()
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


@cli.command()
@click.option(
    "--player_setup", type=click.Choice(list(PLAYER_CONFIGS.keys())), default="HRes"
)
def game(player_setup: str):
    """
    Starts a game with the given player setup
    """
    players = PLAYER_CONFIGS[player_setup]
    if len(players) == 0:
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
                player_type = input(
                    f"Player {i + 1} type? ({'/'.join(options)})\n"
                ).strip()
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

    game = GameManager(players)

    game.start_game()


if __name__ == "__main__":
    cli()
