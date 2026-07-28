from typing import Literal

from datasets import Dataset, load_dataset


def load_imdb(
    split: Literal["test", "train"] = "test", limit: int | None = None
) -> Dataset:
    """
    Load the Stanford IMDb dataset from HuggingFace datasets.

    Parameters
    ----------
    split : Literal["test", "train"]
        Which split of the dataset to load.

    limit : int, Optional
        The maximum number of rows to return. If `None` or missing, then the full
        dataset is returned.

    Returns
    -------
    Dataset
        The requested IMDb dataset split.
    """
    imdb_dataset = load_dataset("stanfordnlp/imdb", split=split)

    if limit is not None:
        imdb_dataset = imdb_dataset.select(range(limit))
    return imdb_dataset
