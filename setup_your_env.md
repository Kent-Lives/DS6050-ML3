# Set Up the DS 6050 Environment

Use the course Conda environment when running assignments from a terminal on Rivanna. This gives everyone the same Python version and package versions.

The setup has two parts:

1. `miniforge/26.9.4` provides Conda and Mamba.
2. [`environment.yml`](environment.yml) defines the Python packages used by the course.

Loading Miniforge alone is not enough. You must also create and activate the `ds6050` environment.

> Do not install course packages with `pip install --user`, and do not add packages to the Miniforge base environment.

## First-time setup

Complete these steps once.

### 1. Start a standard interactive job

Do not perform environment installation or other resource-intensive work on a Rivanna login node.

```bash
ijob -J ds6050_setup \
  -A shakeri_ds6050 \
  -p standard \
  -c 2 \
  -t 02:00:00 \
  --mem=16G \
  -v
```

Wait until the job starts before continuing.

### 2. Load the standard Miniforge module

```bash
module purge
module load miniforge/26.9.4
```

### 3. Create the course environment

Change to the root of your copy of the student repository. This is the directory containing `environment.yml`.

```bash
cd /path/to/deep-learning-student-materials
mamba env create -f environment.yml
```

Creating the environment can take several minutes. You only need to do this once.

### 4. Activate and verify the environment

```bash
conda activate ds6050
python --version
```

The Python version should begin with `3.12`.

Run the dependency check:

```bash
python -c "import numpy, pandas, sklearn, yaml, PIL, torch, torchvision, matplotlib, seaborn, scipy, tqdm; print('DS 6050 environment: PASS'); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available: False` is normal in a standard CPU job. It should be `True` when you request a compatible GPU job.

## Every new Rivanna session

You do not need to recreate the environment. Load Miniforge and activate the existing environment:

```bash
module purge
module load miniforge/26.9.4
conda activate ds6050
```

Confirm that the correct Python is active:

```bash
which python
python --version
```

Then change to an assignment directory and run its runner:

```bash
cd /path/to/deep-learning-student-materials/Assignment_3
python runner.py
```

Run each assignment from its own directory so that Python can find the required assignment files.

## Running assignments with a GPU

Assignments that train neural networks should be run in a GPU job. Request the job before loading and activating the environment:

```bash
ijob -J ds6050_assignment \
  -A shakeri_ds6050 \
  -p gpu \
  --gres=gpu:1 \
  -c 4 \
  -t 06:00:00 \
  --mem=64G \
  -v
```

After the job starts:

```bash
module purge
module load miniforge/26.9.4
conda activate ds6050
```

Check that PyTorch can see the GPU:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"
```

Then run the assignment normally:

```bash
cd /path/to/deep-learning-student-materials/Assignment_3
python runner.py
```

## Updating the environment

The course may add dependencies for later assignments. After downloading an updated `environment.yml`, run:

```bash
module purge
module load miniforge/26.9.4
mamba env update -n ds6050 -f environment.yml --prune
conda activate ds6050
```

The `--prune` option removes packages that are no longer part of the course environment.

Run the dependency check again after an update.

## Troubleshooting

### `conda: command not found`

Load Miniforge first:

```bash
module purge
module load miniforge/26.9.4
```

### `EnvironmentNameNotFound: ds6050`

The environment has not been created. Return to the repository root and run:

```bash
mamba env create -f environment.yml
```

### `ModuleNotFoundError`

Verify that `ds6050` is active:

```bash
conda activate ds6050
which python
```

If the course environment file has changed, update the environment using the instructions above.

### CUDA is unavailable

First confirm that you requested a GPU job rather than a standard job. Then run:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

If `nvidia-smi` cannot see a GPU, start a new GPU job. If `nvidia-smi` works but PyTorch still reports `False`, contact the course staff and include the output of both commands.

## Additional Rivanna help

See [`Accessing_Rivanna.md`](Accessing_Rivanna.md) for instructions on logging in, using Open OnDemand, and connecting through Visual Studio Code.
