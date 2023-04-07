from shallowstack.game.poker_game import PLAYER_CONFIGS, GameManager
from shallowstack.neural_net.datamodule import PokerDataModule
from shallowstack.neural_net.neural_net_trainer import NNTrainer
from shallowstack.player.resolve_player import ResolvePlayer
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
    "--override_turn/--no-override_turn",
    default=False,
    type=click.BOOL,
)
def train_all(epochs: int, data_size: int, override_turn: bool):
    nn_trainer = NNTrainer()
    nn_trainer.train_all_networks(epochs, data_size, override_turn)


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
    game = GameManager(players)

    game.start_game()


if __name__ == "__main__":
    cli()
