import torch
import numpy as np
from typing import List

from shallowstack.poker.card import Card


def create_input_vector(
    r1: np.ndarray, r2: np.ndarray, public_cards: List[Card], pot: int
) -> torch.Tensor:
    r1_t = torch.Tensor(r1).reshape(1, -1)
    r2_t = torch.Tensor(r2).reshape(1, -1)
    public_cards_t = torch.Tensor([card.id for card in public_cards]).reshape(1, -1)
    pot_t = torch.Tensor([pot]).reshape(1, -1)

    return torch.cat([r1_t, r2_t, public_cards_t, pot_t], dim=1)


def create_output_vector(
    v1: torch.Tensor, v2: torch.Tensor, dot_sum: torch.Tensor
) -> torch.Tensor:
    return torch.cat([v1, v2, dot_sum], dim=1)
