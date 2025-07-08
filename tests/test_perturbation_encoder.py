import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from perturbation_encoder import PerturbationEncoder


def test_rxrx1_embedding():
    encoder = PerturbationEncoder(
        dataset_id="HUVEC_small",
        model_type="conditional",
        model_name="SD",
    )
    emb = encoder.get_perturbation_embedding("ID_001")

    assert isinstance(emb, torch.Tensor)
    assert emb.shape == (1, 512, 2048)

    expected = torch.tensor([0.1, 0.2, 0.3], dtype=emb.dtype)
    expected = F.pad(expected, (1022, 1023), value=1)
    expected = expected.repeat(1, 512, 1)

    assert torch.allclose(emb, expected)
