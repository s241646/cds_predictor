import pytest
from cds_repository.data import get_dataloaders
from tests.utils import make_position_encoded_csv


@pytest.fixture(scope="session")
def processed_dataloaders(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("processed_data").joinpath("training")
    data_dir.mkdir(parents=True, exist_ok=True)

    make_position_encoded_csv(data_dir / "train.csv.gz", n=256)
    make_position_encoded_csv(data_dir / "val.csv.gz", n=64)

    train_loader, val_loader = get_dataloaders(
        data_path=data_dir,
        batch_size=32,
        num_workers=0,
    )

    return train_loader, val_loader
