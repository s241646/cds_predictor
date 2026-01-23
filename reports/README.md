# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your
    `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [x] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [x] Consider running a hyperparameter optimization sweep (M14)
* [x] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [x] Add a continues workflow that triggers when data changes (M19)
* [x] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [x] Write API tests for your application and setup continues integration for these (M24)
* [x] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [x] Create a frontend for your API (M26)

### Week 3

* [x] Check how robust your model is towards data drifting (M27)
* [x] Setup collection of input-output data from your deployed application (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [x] Instrument your API with a couple of system metrics (M28)
* [x] Setup cloud monitoring of your instrumented application (M28)
* [x] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group 42

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s241646, s184339, s253771, s260422

### Question 3
> **Did you end up using any open-source frameworks/packages not covered in the course during your project? If so**
> **which did you use and how did they help you complete the project?**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

--- question 3 fill here ---

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We used **uv** for managing our project dependencies. All required packages and their versions are tracked in the **uv.lock** file. To set up a complete copy of our development environment, the following steps should be followed:
1. Clone this repository.
2. Install **uv** (Follow this link: https://docs.astral.sh/uv/getting-started/installation/).
3. Run **uv sync**. This command reads the **uv.lock** file and installs all dependencies as specified, ensuring an identical environment.

When adding or updating packages during the development, we use **uv add <package>**, which updates both the **uv.lock** file and the **pyproject.toml** file. The updated files were then pushed to a branch and merged with main.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We initialized our project using the cookiecutter template specifically developed for this course. From the template, we have filled out the following folders:
- ./.github/workflows (contains our continuous integration workflow files)
- ./configs (stores the optimized hyperparameters)
- ./data folder (stores both raw and processed datasets, as well as temporary files generated when making predictions)
- ./dockerfiles (contains Dockerfiles for training and API containers),
- ./models (the folder is empty, but it is used for saving trained models when running locally)
- ./reports
- ./src/cds_repository (contains all our python scripts)
- ./tests (contains all our unit test python scripts)

We have removed the  ./docs and ./notebooks folder because we did not create documentation or use jupyter notebooks in our project.

We have added the following folders: ./.dvc (contains DVC configuration for data version control), ./deploy (contains documentation on deployment setup), and ./sweeps (contains a configuration file for hyperparameter search space).

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We use pre-commit hooks to automatically run checks before code is committed, ensuring consistent standards across the team during the project. For linting and code quality, we use Ruff, which helps catch common bugs, unused code, and style issues. For formatting, we used on Ruff Format, which ensures consistent code style with formatting rules which are enforced via pre-commit-hooks, for example removing trailing whitespace and ensuring files end with a newlines.

Ruff provides lightweight type checks that improve code correctness. Documentation is supported through well-structured README files, which explain the project structure and usage.

These rules and formalities matter in larger projects because many developers work on the same codebase over long periods of time. Consistent formatting and linting reduce code conflicts and bugs, and good documentation makes it easier for new developers to understand the codebase, contribute and improves maintainability.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total, we have implemented  **24 tests** covering the data, model, training, and API logics.
The data tests validate raw and processed datasets, including CSV consistency and length, correct one-hot encoding, shape, and to validate the train/test/val splits. The model tests ensure MotifCNN and its Lightning wrapper handle input shapes correctly, raise errors on invalid inputs, and compute outputs and accuracy as expected. Training-related tests verify that the training step and optimizer configuration work as expected. Finally, for the API, we implemented integration tests using FastAPI's TestClient to verify endpoint routing, health checks, and the prediction pipeline.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage can be calculated manually by running:
```
uv run coverage run --omit="*/_remote_module_non_scriptable.py" -m pytest tests/
uv run coverage report -m > reports/coverage.txt
```
, and also is automatically calculated when pull requests are made to main. The coverage report is available as an artifact and linked as URL in the workflow run under 'Upload coverage artifact'.

The total code coverage of our code is 86%, which includes all our source files. Coverage is high overall, but some gaps remain, particularly in data.py and model.py, where certain branches and edge cases are not tested. While high coverage increases confidence in the correctness of the code, even 100% coverage would not guarantee it is completely error-free. Code coverage only measures which lines are executed during tests, not whether the logic is correct and bugs can still exist. If the tests do not cover all edge cases, the coverage is also irrepresentative of the code's performance. Therefore, while coverage is a valuable metric, it should be complemented with code reviews and further testing to ensure robust and reliable code.

##REWRITE AFTER API

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We used branches and PRs in our project. In our group, we used branches for adding new features. Each feature was developed separately on a branch, instead of the main branch. Most of the time one member worked on a branch, but other times a member could easily pick up another one's branch.

Once a feature or fix was complete, we created PRs to merge to main, with a short description of the changes / new feature. Most of the time other members reviewed the PRs (sometimes in person), to catch any issues before affecting other code. For changes to the README we directly committed to main. Overall, using branches and PRs helped keep our codebase organized and reduced merge conflicts, as well as be able to trace back to early versions, if needed.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used DVC to manage data. The `data/` folder is tracked with DVC and stored in a GCP bucket. Git only stores the small `data.dvc` pointer file, which records hashes of the data. When someone runs `dvc pull`, the exact data version is downloaded from the bucket. This helped us keep data consistent across machines and to avoid pushing large files to GitHub. It also made the CI workflow possible, because the data can be pulled during a workflow run. If we update data, we run `dvc add data` and `dvc push`, then commit the new `data.dvc`. It also gives us a clear history of data changes and rollbacks.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

Our continuous integration setup can be seen in ```./.github/workflows/```. We have organized it into 7 separate files, as described below:
- Unit testing with pytest (```tests.yaml```): We have defined multiple unit tests (see ```./tests/```), which we test on both macOS, Ubuntu, and Windows. For all operative systems, we test Python versions 3.12 and 3.13. In addition, we test different PyTorch versions (2.5.0, 2.6.0, 2.7.0) on Ubuntu with both Python versions 3.12 and 3.13 to ensure compatability across our supported PyTorch range.
- Code coverage calculation on Ubuntu with Python 3.12 (```coverage.yaml```): We calculate the code coverage every time something is pushed to the repository. This is not necessary, but gives as an easy way to check the coverage at any time something is modified. ... **REMOVE?**
- Code linting check with ruff (```linting.yaml```): We check that all scripts passes the format checks with ruff, following the standards specified in ./pyproject.toml.
- Automated pre-commit hook updates (```pre-commit-update.yaml```): Since the ```.pre-commit-config.yaml``` file is not automatically updated by Dependabot, we've created ```pre-commit-update.yaml```, that automatically updates the workflow at midnight and creates a PR if there are any changes to the pre-commit hooks. The workflow can be found here: https://github.com/s241646/cds_predictor/actions/workflows/pre-commit-update.yaml, and uses https://github.com/peter-evans/create-pull-request.
- Check proper data loading after updates (```data-change.yml```): Checks that data can still be pulled and loaded properly, after changing data-related files.
- Check model registry change (```model-registry-change.yml```): Triggered when model files change to verify that model artifacts exist in the ./models folder.
- (```deploy-cloud-run.yml```): **ADD HERE**

We make use of caching, though this is not explicitly visible in the workflow .yaml files. The setup-uv action (astral-sh/setup-uv@v7) has built-in caching that stores both the uv tool itself and the installed Python packages, speeding up our continuous integration runs.

An example of a triggered workflow can be seen here: https://github.com/s241646/cds_predictor/blob/main/.github/workflows/tests.yaml



## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We configured experiments by using Hydra config files in the train script. The optimized hyperparameters (batch size, learning rate, scheduler, architecture) are defined in configs/config.yaml, and the training script loads this configuration using @hydra.main(). The parameters can be overwritten from the command line, for example as:
```
python -m cds_repository.train epochs=50 hyperparameters.batch_size=32 dropout=0.3
```

The parameters can also be integrated with hyperparameter sweeps, using the W&B sweep configurations specified in sweeps/motifcnn_sweep.yaml.


### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

The hydra config files were version-controlled (git) and logged to the timestamped folder. Each log stores the exact configuration used. We used fixed random seeds everywhere for reproducibility, which is of course very important (we have set the seed to 42). Each experiments' hyperparameters, metrics, and best model artifacts are tracked in Weights & Biases. Since everyone in the team has access to the same Weights and Biases prject, everyone can access all experiments run from anyone else in the team. To rerun past experiments (with access to the W&B team), the following command can be run:
```
python train.py +wandb_run_id=<run_id>
```
, or the config file can be reused (./configs/config.yaml). 

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

We tracked and visualized experiments in Weights & Biases, as well as analyzing hyperparameter sweeps.

During training, we track both the loss and accuracy on both the training and validation set. Monitoring of these metrics helps us monitor points of model convergence, potential overfitting and generalization. A model that does well should decrease validation loss, have increasing validation accuracy, and ideally similar training and validation loss when averaged over batches. We could also have logged additional metrics such as AUC, sensitivity and specificity (these would also be useful to monitor since the dataset is imbalanced).
![Logging of experiments with loss curves and performance progress in Weights and Biases.](./figures/Q14_wandb_log_progress.png)

The next image shows the parameter importance of the most useful/important parameters based on training accuracy.
![Screenshot showing which hyperparameters are most important for training accuracy in Weights and Biases.](./figures/Q14_hyperparameter_importance.png)
From our sweeps, we could infer that learning rate is strongly positvely correlated with train accuracy. Dropout is also important, but has a negative correlation, showcasing that lower dropout rates are preferred. This was useful in priotizing impactful hyperparameters.

![Saving of model checkpoints in Weights and Biases.](./figures/Q14_model_checkpoints.png)
The image above shows the saving of model checkpoints, which are ready to be reused in the API for example. Artifacts are linked to the model registry

Together, these metrics and visualizations provide a comprehensive view of model performance, stability, and reproducibility across experiments.

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

We use Docker for both training and inference so the environment is identical across machines. The Dockerfiles live in `dockerfiles/` (for example `dockerfiles/api.dockerfile`, `dockerfiles/train.dockerfile`, and `dockerfiles/frontend.dockerfile`). For local API usage we build and run:

`docker build -f dockerfiles/api.dockerfile -t cds-api .`

`docker run --rm -p 8000:8000 cds-api`

For training, we use the training image and pass Hydra overrides:

`docker build -f dockerfiles/train.dockerfile -t cds-train .`

`docker run --rm cds-train epochs=5`

The training container expects the dataset to be available locally or pulled with DVC at runtime, which avoids baking large data into the image. The API image is also used for Cloud Run deployment, so the container used in development matches production. This setup reduces issues linked to running in different machines. Link to Dockerfile:
`dockerfiles/api.dockerfile`.

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

When running into bugs, we mostly asked an LLM to help us understand the bug in order to fix it.

We did profile the code, and below is a screenshot of the output report:

![Logging of experiments with loss curves and performance progress in Weights and Biases.](./figures/Q16_profiler_results.jpeg)

One training epoch only took about 30 seconds which we believe is already quite fast. For this reason, we did not try to optimize our code based on the profiling. We think that code can always be optimized somehow (and hence it is not perfect), but for this project we decided to prioritize our resoruces elsewhere.


## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used the following services: Cloud Storage, Artifact Registry, Cloud Run, Cloud Build, Cloud Monitoring, Cloud Logging.
- Cloud Storage is used as the DVC remote for datasets.
- Artifact Registry stores our Docker images (API and training).
- Cloud Run hosts the deployed API and provides autoscaling.
- Cloud Build uses `cloudbuild.yaml` to build and push a training image when triggered.
- Cloud Monitoring provides built-in Cloud Run metrics and alerting for error rate and latency.
- Cloud Logging enables inspection of custom train jobs logs
This combination covers data storage, container hosting, CI builds, and service monitoring.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used Google Compute Engine for running and managing our training infrastructure. Specifically, we created a VM instance ```cds-instance``` for launching and controlling our training jobs.

We start the VMM and connect via SSH after authenticating with ```gcloud```:
```
gcloud compute instances start cds-instance

gcloud compute ssh cds-instance
```

Once connected, we clone our repository, install dependencies, and configure the environment. From this VM, we triggered Vertex AI custom training jobs using the ```gcloud ai custom-jobs create command```. Specifically:
```
gcloud ai custom-jobs create \
    --region=europe-west1 \
    --display-name=train-run \
    --config=config_cpu.yaml \
```

While the actual training is orchestrated by Vertex AI, Compute Engine provides the compute resources and a stable control point for managing experiments.

The VM we used was an e2-standard-4 instance (4 vCPUs, 16 GB RAM) located in europe-west1-b, with a 100 GB persistent disk. This configuration provided sufficient memory, CPU performance, and storage capacity for heavy preprocessing, model training, and saving checkpoints. The persistent disk allowed us to store intermediate artifacts and model checkpoints reliably during long-running training jobs.

View the progress at GCP: Vertex AI > Model development > Training > Custom jobs
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=cds-predictor


### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---


![Bucket outputs.](./figures/Q19_bucket1.png)
![Bucket models.](./figures/Q19_bucket2.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---
![Artifacts.](./figures/Q20_artifacts.png)


### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![Artifacts.](./figures/Q21_cloudbuild.png)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We managed to train our model in the cloud using the Compute Engine. We did this by creating a VM instance with sufficient CPU, memory and disk capacity, with Pytorch already installed. We starting it by the terminal or in GCP, and connecting via SSH. The VM was useful because we could use Vertex Ai's logging to track the status of the training job, and intermediate checkpoints and outputs are saved to storage. This set-up enabled reliable and scalable model training, without relying on local hardware. We could submit a job and check back in after a few hours, without worrying about relying on and using resources of our own devices.

Example seeing jobs on VM:
![Artifacts.](./figures/Q21_VM.png)

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

Yes, we built an API using FastAPI in `src/cds_repository/api.py`. The API loads a trained checkpoint and exposes lightweight HTTP endpoints for inference. We provide `/health` for status checks, `/` for a basic service message, and `/predict` for inference. The `/predict` endpoint accepts a DNA sequence (and in the extended version can accept FASTA input) and returns logits and a probability score so the client can interpret the model output. We also keep the inputs small and validate them to avoid unnecessary load or failures. This API is used both locally and in Cloud Run so the same interface works in development and production. Overall, the API wraps the model in a simple, repeatable interface that other services and teammates can call without needing to run training code.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

Yes, we deployed the API to GCP Cloud Run. The GitHub Actions workflow (`.github/workflows/deploy-cloud-run.yml`) builds the Docker image from `dockerfiles/api.dockerfile`, pushes it to Artifact Registry, and deploys it to a Cloud Run service in `europe-west1`. The service is configured to listen on port 8000 and is publicly accessible over HTTPS. We validate the deployment by calling `/health`, `/`, and `/predict` endpoints from a local terminal. Example usage:

`curl https://cds-api-978941563399.europe-west1.run.app/health`

`curl -s -X POST https://cds-api-978941563399.europe-west1.run.app/predict -H "Content-Type: application/json" -d '{"sequence":"ATGCGT"}'`

This confirms the API is reachable and serving predictions in the cloud, and it also documents how a user would invoke the service after deployment.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

For unit and integration testing, we implemented a comprehensive testing framework using pytest and FastAPI's TestClient. Since our production model is stored in a GCS bucket that GitHub Actions cannot access, we also implemented a conftest.py file to mock the model inference and storage layers. To find out where our service actually breaks, we conducted a local load test using Locust, ramping up concurrent users to empirically identify our hardware's saturation point.

Our testing identified a clear CPU-bound bottleneck during CNN inference. At 10 concurrent users, the API reached its most efficient state with a median prediction latency of 3.1 seconds and a 95th percentile of 6.9 seconds. As we scaled to 20 users, throughput (RPS) remained flat while latency doubled, indicating the system had reached its maximum work capacity. Beyond this point, the service became severely congested: at 40 users, 95th percentile response times spiked to 39 seconds, and at 50 users, the system failed with a 7% error rate. We concluded that the current configuration can reliably support 10 concurrent users with acceptable latency, but horizontal scaling or request batching would be mandatory to support larger volumes.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We used Cloud Monitoring for our Cloud Run service. Cloud Run provides built-in metrics such as request count, latency, and error rate, which we tracked in the Monitoring dashboards. We also instrumented the API with a `/metrics` endpoint that reports request counts, error counts, latency, and basic system info, so we can inspect health even without the GCP console. We created alert policies for 5xx error rate and high latency (p95), and we verified alerts by triggering a request-count alert and receiving an email notification. This gives us both proactive alerts and ongoing visibility into service behavior and performance. For evidence, we exported the alert policies to `monitoring/alert-policies.json`.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- question 27 fill here ---
The largest cost to the team was the Compute Engine, followed by Vertex AI, due to the training jobs that were ran there.
**TODO** report on actual costs and cost of api / cloud run later

Working on the cloud was frustrating, because it took a while for all the setups between data, model, training, API, inputs/outputs to work. However, once the connections were established it was useful to run long training jobs, and we experienced it was faster to make predictions on the API with the ckpt from the GCP bucket, versus the local checkpoint. It was frustrating at times to manage permission roles, especially when uploading files to the GCP buckets.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

We checked how robust our model is to data drifting. As an experiment, we checked for data drift on the input data, between the training set and the test set. The genomes in the training and test sets use the same genetic code (translation table 11). However, for real-world applications we would expect that the method might also be used for sequences that originate from genomes that use alternative genetic codes. For this reason, we also checked for data drift on a genome that uses translation table 4. The script ```src/cds_repository/data_drift.py``` calculates data drift between the training data set and a dataset specified via terminal. It generates a report based both on input features and based on the final sequence representation (embeddings) before the output layer. The reports generated with Evidently can be seen in ```./reports/drift_check/```. The drift check can also be ran on uploaded data from the API. See README for useage.


![Drift.](./figures/Q28_drift.png)
![Drift2.](./figures/Q28_drift2.png)

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

*see reports/mlops_diagram.pdf for high resolution diagram**

![Diagram.](./figures/mlops_diagram.png)

The starting point of the diagram is our GitHub Repository, which should be clones to begin with. When code is pushed to main (or a PR is created), several GitHub Actions run automatically. Some GitHub actions (pre-commit auto-update) run periodically, each day at midnight. When changes are made to the train / api files, the docker image is automatically rebuilt and pushed to the Artifact Registry, tagged with 'latest'. The latest image is deployed via cloud run (both backend and frontend).

To use the ML model, one can either launch the inference API locally or by accessing the link: https://streamlit-frontend-978941563399.europe-west1.run.app/. The Frontend is created with Streamlit. The user can upload a FASTA file, which is converted first to a csv with valid column headers, and then to one-hot encodings. The model returns 1/0. The user's input csv file and the models output are registered in the GCP bucket. The API has a /metrics endpoint to monitor the behaviour.

In the 'storage' part of the diagram, the link between the DVC and local storage is shown, wich mutually track changes to data.

Pytorch Lightning - based training can either be done locally, or via the Compute Engine VM instance. Both should use data pulled from the DVC, and preprocessed by one-hot encoding. When training on the Compute Engine, logs can be observed in Cloud Logging (Observability), and logs are written to Hydra as well. For local trainingn, WandB logging and agent sweeping is also available, for hyperparameter tuning and importance analysis.

In the right-most box, the remaining logging and monitoring modules of the pipeline are shown. Data drift detection can be done over local data or the saved inputs from the API.


### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

The biggest challenges in this project were related to solving merge conflicts, both on GitHub and with DVC. When developing a new functionality, we practiced creating a new branch, pushing to it, and then merging with main, which definitely helped avoiding major merge conflicts. However, sometimes more than one person would work on the same functionality etc. and solving merge comflicts from this was challenging at times.

Apart from the merge conflicts, the deployment also took quite some time. It was not because there was any big struggles related to this part, but it was just one of the more time consuming tasks because none of us had any experience with it. For this reason, it took more time to figure out the cause of each bug. At first it was difficult because we did not know where and how to look for the bugs and solutions to them, but at the end of week 2 we became more familiar with it, and it definitely helped to gain some confidence with working on the cloud.

Furthermore, we also spent a lot of time connecting everything together and making sure that all functionalities were up to date when implementing new things.

Overall, we had a good group dynamic and helped each other out, so that challenges did not become too big to overcome for us.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

All members of the team contributed equally, but worked mostly on the following: 
- s241646: Created the MLOps Diagram, setup upGCP model artifacts logging, cloud build and model training on Compute Engine VM. Making data drift checking compatible with API output in GCP bucket. Some docker configurations, pre-commit autoupdate fixes and setting up WandB team and logging.
- s184339: Generated the datasets and project description, preprocessing of data for application, set up the continuous integration workflow and pre-commit, implemented data drift checking. 
- s253771: Built the model and training setup, maintained the Dockerfiles for training/deployment, and set up DVC with GCS for data versioning. Also handled deployment/monitoring setup and updated documentation.
- s260422: Developed the FastAPI application and frontend, including input-output data collection. Refactored training using PyTorch Lightning, integrated Weights & Biases for hyperparameter sweeps, and run profiling. Also established the API testing suite with CI mocking and performed local load testing to identify bottlenecks.

We always met up in the team, and discussed the different aspects of the tasks we were each doing, as well as to collaborate on a large part of the tasks. All members reviewed each other's work and contributed to debugging, README updates, report writing, and pipeline design. In this way we ensured that all team members had an understanding of the whole project. 

We used generative AI to debug errors, draft workflow steps, and improve documentation. We always verified changes by running commands and checking logs.
