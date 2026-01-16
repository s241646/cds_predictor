# CDS predictor in short prokaryotic DNA sequences

## Project description
Identification of protein-coding genes and their corresponding coding regions is a well-established task in prokaryotic genomes. Genomic data is typically sequenced into many short DNA fragments called reads. Various methods exist to assemble these reads into longer sequences and to align them to known genomic references. However, assembly is not always possible, requiring alternative approaches to determine whether a sequence is protein-coding and thus functionally relevant. This is frequently the case in metagenomics samples or ancient DNA.

The overall goal of this project is to develop and train a model that can predict whether a short DNA sequence belongs to a protein-coding gene or not. Therefore, we frame this as a binary classification task.

While more complex problem definitions are required to make it useful for real-world scenarios, this project scope is a step in that direction.

The dataset we use is a simplified subset of a dataset curated by a group member for another project. The simplified dataset used for this project is placed in ```/data/raw/```. Input sequences are short DNA fragments of 300 nucleotides, with a vocabulary of {A, T, G, C}, labeled as 0 (non-coding) or 1 (coding).

The dataset includes sequences from 8 bacterial genomes. Sequences are partiotioned so that 4 genomes are used for the training set, 2 genomes for the validation set, and 2 genomes for the test set. This ensures that no sequences from the same genome appear in different partitions. One could also split the sequences based on sequence similarity, but this approach is computationally intensive and not related to the scope of the MLOps course.

We are not completely sure what model framework we will end up using, but our plan is to one-hot encode sequences and start with a simple CNN-based architecture.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Set-up
Make sure you have the uv package manager and uvx for tool invocations are installed
```
uv --version
```

Install and sync dependencies:
```
uv sync
```

Processed data in data/processed is created by running (already done)
```
uvx invoke preprocess-data
```

## Key Commands

### Package Management
```bash
uv add <package-name>          # Install a package
uv sync                        # Sync dependencies with pyproject.toml
```

### Running Code
```bash
uv run <script-name>.py        # Run a Python script
uv run invoke --list           # List available tasks
uv run invoke <task-name>      # Run a specific task
```

### Testing & Code Quality
```bash
uv run pytest tests/           # Run all tests
uv run ruff format .           # Format code
uv run ruff check . --fix      # Lint and fix issues
```
Pre-commit hooks:
```
uv run pre-commit install
uv run pre-commit run --all-files  # Run pre-commit hooks
```

### Documentation
```bash
uv run mkdocs serve            # Build and serve docs locally
```


### Wandb
Create a file .env, add the following:
```
WANDB_API_KEY=""
WANDB_PROJECT=cds_predictor
WANDB_ENTITY=mlops_group42
```

## Unit-tests
```uv run pytest tests```

## Coverage report
```
coverage run --omit="*/_remote_module_non_scriptable.py" -m pytest
coverage report -m
```

## Autoupdate workflow files manually
Github Actions cannot autoupdate workflow files. Run the following to do it manually:
```
pre-commit autoupdate
git checkout -b pre-commit-autoupdate-xxxx
git commit -am "chore: pre-commit autoupdate"
git push origin pre-commit-autoupdate-xxxx
```
