"""Data preprocessing utilities for the GAN model."""

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_and_preprocess_data(csv_path: str, batch_size: int = 128) -> DataLoader:
    """Load Kaggle digit-recognizer data and normalize pixels to [-1, 1]."""
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError(
            "Expected a Kaggle digit-recognizer style CSV with a 'label' column."
        )

    images = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)
    images = (images / 127.5) - 1.0

    return DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=True)
