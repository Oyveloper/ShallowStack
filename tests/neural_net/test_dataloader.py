from shallowstack.neural_net.datamodule import PokerDataModule
from shallowstack.state_manager.state_manager import PokerGameStage


def test_load_example():
    data = PokerDataModule(PokerGameStage.RIVER, 1)
    data.setup("")

    i = iter(data.train_dataloader())
    print(next(i))
