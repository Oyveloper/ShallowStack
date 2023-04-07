import os
from glob import glob

from shallowstack.neural_net.model import ValueNetwork
from shallowstack.state_manager.state_manager import PokerGameStage


class NNManager:
    def __init__(self):
        # Load the networks
        self.river_network = self.load_network(PokerGameStage.RIVER)
        self.turn_network = self.load_network(PokerGameStage.TURN)
        self.flop_network = self.load_network(PokerGameStage.FLOP)
        self.pre_flop_network = self.load_network(PokerGameStage.PRE_FLOP)

    def get_network(self, stage: PokerGameStage) -> ValueNetwork:
        """
        Returns the network for a given stage
        """
        network = self.pre_flop_network

        if stage == PokerGameStage.FLOP:
            network = self.flop_network
        elif stage == PokerGameStage.TURN:
            network = self.turn_network
        elif stage == PokerGameStage.RIVER:
            network = self.river_network

        return network

    def load_network(self, stage: PokerGameStage, version: int = -1) -> ValueNetwork:
        nbr_public_cards = 0
        if stage == PokerGameStage.FLOP:
            nbr_public_cards = 3
        elif stage == PokerGameStage.TURN:
            nbr_public_cards = 4
        elif stage == PokerGameStage.RIVER:
            nbr_public_cards = 5

        network = ValueNetwork(1326, nbr_public_cards)

        try:
            stage_dir = f"lightning_logs/{stage.name}/lightning_logs/"
            folders = glob(stage_dir + "version_*")
            if len(folders) == 0:
                raise Exception("No folders for network")

            sorted_dirs = sorted(folders, key=os.path.getmtime)

            dir = sorted_dirs[version]

            f = glob(f"{dir}/checkpoints/*.ckpt")[0]
            network.load_from_checkpoint(f)
        except:
            pass

        return network
