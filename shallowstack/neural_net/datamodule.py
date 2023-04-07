from multiprocessing import Pool
import os
import shutil
from typing import List
from typing import Optional
from typing import Tuple

import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
from shallowstack.game.action import AGENT_ACTIONS

from shallowstack.neural_net.util import create_input_vector, create_output_vector
from shallowstack.player.player import Player
from shallowstack.poker.card import Card
from shallowstack.poker.card import Deck
from shallowstack.state_manager.state_manager import GameState, PokerGameStage
from shallowstack.subtree.subtree_manager import AVG_POT_SIZE
from shallowstack.subtree.subtree_manager import SubtreeManager


def generate_random_ranges(public_cards: List[Card]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates two player ranges filtered by the provdied public cards
    """
    r1 = np.random.random(1326)
    r1 = r1 / np.sum(r1)
    r2 = np.random.random(1326)
    r2 = r2 / np.sum(r2)

    r1 = SubtreeManager.update_range_from_public_cards(r1, public_cards)
    r2 = SubtreeManager.update_range_from_public_cards(r2, public_cards)

    return (r1, r2)


def generate_initial_situation_from_public_cards(
    public_cards: List[Card],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generates one random startin_value for neural net training

    """
    r1, r2 = generate_random_ranges(public_cards)
    pot = np.random.randint(0, AVG_POT_SIZE)

    return (r1, r2, pot)


def get_calculated_values_for_situation(
    stage: PokerGameStage,
    r1: np.ndarray,
    r2: np.ndarray,
    pot: int,
    public_cards: List[Card],
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the calculated values for the given situation"""
    deck = Deck()
    deck.remove_cards(public_cards)

    game_state = GameState(
        stage,
        [
            Player("Player 1", 0),
            Player("Player 2", 0),
        ],
        0,
        np.array([pot / 2, pot / 2]),
        np.zeros(2),
        np.ones(2),
        pot,
        pot // 2,
        public_cards,
        deck,
    )

    end_stage = stage
    end_depth = 1

    if stage == PokerGameStage.PRE_FLOP:
        end_stage = PokerGameStage.FLOP
    elif stage == PokerGameStage.FLOP:
        end_stage = PokerGameStage.TURN
    elif stage == PokerGameStage.TURN:
        end_stage = PokerGameStage.RIVER
    elif stage == PokerGameStage.RIVER:
        end_depth = 20

    strategy = np.ones((1326, len(AGENT_ACTIONS))) / 1326

    tree = SubtreeManager(game_state, end_stage, end_depth, strategy)

    v1, v2 = tree.subtree_traversal_rollout(tree.root, r1, r2)

    return (v1, v2)


def get_random_example(arg: Tuple[PokerGameStage, int]) -> torch.Tensor:
    stage, nbr_public_cards = arg

    d = Deck()
    public_cards = d.draw(nbr_public_cards)

    r1, r2, pot = generate_initial_situation_from_public_cards(public_cards)

    v1, v2 = get_calculated_values_for_situation(stage, r1, r2, pot, public_cards)

    output = create_output_vector(
        torch.tensor(v1).reshape(1, -1),
        torch.tensor(v2).reshape(1, -1),
        torch.zeros(1).reshape(1, -1),
    )

    example = torch.cat((create_input_vector(r1, r2, public_cards, pot), output), dim=1)

    return example


class PokerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        stage: PokerGameStage,
        batch_size: int,
        data_size: int = 1000,
        force_override: bool = False,
    ):
        super().__init__()
        self.stage = stage
        self.data_dir = f"data/{stage.name}"
        self.batch_size = batch_size
        self.data_size = data_size

        if force_override and os.path.exists(self.data_dir):
            os.chmod(self.data_dir, 0o777)
            shutil.rmtree(self.data_dir, ignore_errors=True)

    def setup(self, stage: str):
        if not os.path.exists(self.data_dir):
            self.generate_dataset(self.stage, self.data_size, show_progress=True)

        self.test_data = torch.load(f"{self.data_dir}/test.pt")
        all = torch.load(f"{self.data_dir}/train.pt")
        all = all.type(torch.float32)

        val_fraction = 0.1

        val_size = int(val_fraction * len(all))
        train_size = len(all) - val_size

        self.train_data, self.val_data = random_split(all, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=4)

    # INternal stuff

    def generate_dataset(
        self,
        stage: PokerGameStage,
        size: int,
        name: Optional[str] = None,
        show_progress: bool = False,
    ):
        """
        Generates a dataset saved to the data folder for the given stage and size

        if name is given this will be used to override the dataset name, if not the name
        is based on the stage and nbr examples
        """
        print(f"generating data for {stage.name}")
        if name is None:
            name = f"{stage.name}"

        nbr_public = 0

        if stage == PokerGameStage.PRE_FLOP:
            nbr_public = 0
        elif stage == PokerGameStage.FLOP:
            nbr_public = 3
        elif stage == PokerGameStage.TURN:
            nbr_public = 4
        elif stage == PokerGameStage.RIVER:
            nbr_public = 5

        test_fraction = 0.2
        nbr_test = int(test_fraction * size)

        with Pool(3) as p:
            dataset = torch.cat(
                list(
                    tqdm(
                        p.imap(get_random_example, [(stage, nbr_public)] * size),
                        total=size,
                        disable=not show_progress,
                    )
                )
            )

        print("Finished with dataset generation")

        train_dataset, test_dataset = dataset.split([size - nbr_test, nbr_test])

        if not os.path.exists(f"data/{name}"):
            os.makedirs(f"data/{name}")
        torch.save(train_dataset, f"data/{name}/train.pt")
        torch.save(test_dataset, f"data/{name}/test.pt")
