import os
import sys
import torch

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
    assert emb.shape == (1, 512, 2048) or emb.shape == (1, 77, 768) or emb.shape[0] == 1
    # Embedding should contain the padded values
    assert torch.all(emb != 0)