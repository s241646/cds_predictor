import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "cds_repository"
PYTHON_VERSION = "3.12"

# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess raw data to one-hot encodings."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)

@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)

@task
def experiment(
    ctx: Context,
    epochs: int = 15,
    lr: float = 3e-4,
    dropout: float = 0.2,
    batch_size: int = 64,
    patience: int = 5,
    channels: str = "64,128,256",
    name: str | None = None,
) -> None:
    """Run a hyperparameter experiment with TensorBoard logging.
    
    Args:
        epochs: Number of epochs to train.
        lr: Learning rate.
        dropout: Dropout probability.
        batch_size: Batch size for training.
        patience: Early stopping patience.
        channels: Comma-separated channel sizes (e.g., '64,128,256').
        name: Experiment name for logging (auto-generated if not provided).
    
    Examples:
        uv run invoke experiment --epochs 20 --lr 1e-4 --channels '128,256,512'
        uv run invoke experiment --epochs 10 --dropout 0.5 --name 'high_dropout'
    """
    # Parse channels
    try:
        channels_list = tuple(int(c.strip()) for c in channels.split(","))
    except ValueError:
        print(f"Error: channels must be comma-separated integers, got '{channels}'")
        return
    
    # Build Python command
    cmd = (
        f"uv run python -c \""
        f"import sys; sys.path.insert(0,'src'); "
        f"from cds_repository.data import get_dataloaders; "
        f"from cds_repository.train import run_training, TrainConfig; "
        f"train_loader, val_loader = get_dataloaders(batch_size={batch_size}); "
        f"cfg=TrainConfig("
        f"epochs={epochs}, "
        f"lr={lr}, "
        f"dropout={dropout}, "
        f"patience={patience}, "
        f"channels={channels_list}"
        f"); "
        f"run_training(train_loader, val_loader, cfg, experiment_name='{name}')"
        f"\""
    )
    
    ctx.run(cmd, echo=True, pty=not WINDOWS)
    if name:
        print(f"\nView results with: tensorboard --logdir logs/{name}")
    else:
        print("\nView results with: tensorboard --logdir logs")

@task
def tensorboard(ctx: Context, logdir: str = "logs") -> None:
    """Launch TensorBoard to view experiment logs.
    
    Args:
        logdir: Directory containing TensorBoard logs (default: logs).
    """
    ctx.run(f"uv run tensorboard --logdir {logdir}", echo=True, pty=not WINDOWS)

@task
def sweep_create(ctx: Context, config_file: str = "configs/sweep.yaml") -> None:
    """Create a wandb hyperparameter optimization sweep.
    
    Args:
        config_file: Path to sweep configuration file (default: configs/sweep.yaml).
    
    Example:
        uv run invoke sweep-create
    """
    ctx.run(f"uv run wandb sweep {config_file}", echo=True, pty=not WINDOWS)

@task
def sweep_agent(ctx: Context, sweep_id: str) -> None:
    """Run a wandb sweep agent for hyperparameter optimization.
    
    Args:
        sweep_id: The sweep ID returned by sweep-create (format: project/sweep_id).
    
    Example:
        uv run invoke sweep-agent --sweep-id "username/cds_predictor/a1b2c3d4"
    """
    ctx.run(
        f"uv run wandb agent {sweep_id}",
        echo=True,
        pty=not WINDOWS
    )

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)

@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
