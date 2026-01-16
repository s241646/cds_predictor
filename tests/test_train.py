import pytorch_lightning as pl
from cds_repository.model import MotifCNNModule
from torch.utils.data import DataLoader, TensorDataset


def test_training_learns_on_processed_data(processed_dataloaders):
    train_loader, val_loader = processed_dataloaders

    model = MotifCNNModule(
        lr=5e-3,
        dropout=0.0,
        use_scheduler=False,
    )

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
    )

    trainer.fit(model, train_loader, val_loader)

    metrics = trainer.callback_metrics

    train_loss = metrics["train_loss_epoch"].item()
    train_acc = metrics["train_acc_epoch"].item()
    val_loss = metrics["val_loss"].item()
    val_acc = metrics["val_acc"].item()

    assert train_loss < 0.7
    assert val_loss < 0.7
    assert train_acc > 0.65
    assert val_acc > 0.65


def test_overfit_single_batch(processed_dataloaders):
    train_loader, _ = processed_dataloaders
    x, y = next(iter(train_loader))

    single_loader = DataLoader(
        TensorDataset(x, y),
        batch_size=len(y),
        shuffle=False,
    )

    model = MotifCNNModule(
        lr=1e-2,
        dropout=0.0,
        use_scheduler=False,
    )

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model, single_loader)

    acc = model._acc(model(x), y).item()
    assert acc > 0.95
