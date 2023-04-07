from lightning import Trainer
from shallowstack.neural_net.datamodule import PokerDataModule
from shallowstack.neural_net.model import ValueNetwork
from shallowstack.state_manager.state_manager import PokerGameStage


class NNTrainer:
    def __init__(self):
        self.training_order = [
            PokerGameStage.RIVER,
            PokerGameStage.TURN,
            PokerGameStage.FLOP,
        ]

    def train_network(
        self,
        stage: PokerGameStage,
        max_epochs: int = 100,
        data_size: int = 100,
        override_data: bool = False,
    ):
        """
        Trains a single network for a given stage, using the data
        of games
        """
        nbr_public_cards = 0
        if stage == PokerGameStage.FLOP:
            nbr_public_cards = 3
        elif stage == PokerGameStage.TURN:
            nbr_public_cards = 4
        elif stage == PokerGameStage.RIVER:
            nbr_public_cards = 5

        network = ValueNetwork(1326, nbr_public_cards)
        data = PokerDataModule(stage, 10, data_size, force_override=override_data)
        trainer = Trainer(
            max_epochs=max_epochs, default_root_dir=f"lightning_logs/{stage.name}"
        )

        trainer.fit(network, data)

    def train_all_networks(
        self, max_ephochs: int = 100, data_size: int = 100, override_river: bool = False
    ):
        """
        Trains networks for all stages in a bottom up manner
        """
        override = override_river
        for stage in self.training_order:
            print(f"Training network: {stage.name}")
            self.train_network(
                stage,
                override_data=override,
                max_epochs=max_ephochs,
                data_size=data_size,
            )

            # Make sure that every network except the first must generate new data
            # Since the lower network is updated
            override = True
