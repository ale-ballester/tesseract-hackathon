# **Plasma Control via Reduced-Order Modeling** (change title as needed)
This repository contains code for simulating and controlling a 1-D particle-in-cell (PIC) plasma system using reduced-order modeling techniques. The main components include a PIC simulator, a linear reduced-order model, functionality for external control inputs, and training routines to fit the ROM to simulation data. 


---
## **Setup and installation**
Ensure you have an appropriate JAX build for your platform. See [JAX installation instructions](https://docs.jax.dev/en/latest/installation.html) for more details.

Minimal packages used in the code:
  - `jax`, `jaxlib`
  - `equinox`
  - `optax`
  - `diffrax`
  - `lineax`
  - `matplotlib`
  - `numpy`
  - `pickle`

Example installation (CPU-only):
```python
pip install "jax[cpu]" equinox optax diffrax lineax matplotlib numpy
```
---
## **Important Files and Structure**
  - `pic_simulation.py`: Contains an implementation of a 1-D particle-in-cell simulator.
    - `PICSimulator`: Equinox module implementing the PIC simulation.
  - `rom.py`: Contains the reduced-order model implementation and utilities for fitting LDS to data.
    - `LDSModel`: Equinox module implementing a linear dynamical system with integration via Diffrax.
    - `fit_discrete_linear_system_ridge`/`fit_ct_trapezoid_ridge`: Utilities for fitting discrete-time and continuous-time linear systems to data.
    - `fom_state_to_rom_state`/`rom_state_to_fom_state`: Utilities for converting between full-order and reduced-order model states.
  - `control.py`: Utilities for generating and using external control inputs, as well as basic LQR functionality.
  - `main_zir.py`: Runs the PIC simulation without any external control input.
  - `main_resp.py`: Runs the PIC simulation with a prescribed control input.
  - `main_opt.py`: Runs the PIC simulation with optimized control input.
  - `main_train.py`: Loads data, instantiates the model, and runs the training.
  - `train.py`: Contains the training loop and utilities for plotting.
  - `data/` and `dataloader.py`: Stores the dataset files and contains utilities for loading and batching the data.
  - `model/`: Contains the saved model checkpoints.
  - `plots/`: Contains the plots generated during the training.


## **Architecture**
### Reduced-Order Model (ROM)
Implemented in `rom.LDSModel`, the ROM is defined as `dx/dt = A x(t) + B u(t)`, where:
- `x(t) in R^(n_out)`
- `u(t) in R^(n_in)`
- `A in R^(n_out x n_out)`
- `B in R^(n_out x n_in)`

In discrete-time, the model is defined as `x[k+1] = A' x[k] + B' u[k]`, where `A'` and `B'` are the discrete-time equivalents of `A` and `B`, respectively.

Can toggle between open-loop and closed-loop modes:
- In open-loop mode, the model uses the input `u(t)` directly from the dataset.
- In closed-loop mode, the model's own output is fed back as input for the next time step.

Integration is performed using Diffrax.

`save_model` and `load_model` methods allow for checkpointing and loading of saved models, respectively.

### Training
The training loop is implemented in `train.Trainer`. The default setup includes:
- Loss: L2 over predicted vs. true trajectories
- Optimizer: `optax.adam`
- Data batching and shuffling using `dataloader.DataLoader`
- The training step is JIT-compiled to accelerate updates

The training process works as follows (on a high level):
- Load the dataset (from `data/` directory or custom path)
- Initialize the ROM (`LDSModel`) appropriately
- Initialize the `Trainer` with the model, data, optimizer, loss function, and other settings
- Run the training loop, periodically checkpointing and plotting

## **Usage**
### Examples
To use the given default training script, the `data/` directory should contain `dataset.pkl`, `modes.pkl`, and `params.pkl`, with the data formatted appropriately. Then, run:
```python
python main_train.py
```
To customize the model or training process, the following can be directly adjusted in `main_train.py`:
- **Data Loading**: Change how the dataset is loaded and preprocessed.
- **Model Initialization**: Change the model architecture, parameters, or hyperparameters.
- **Training**: Change the loss function, optimizer, or other training settings.

To load a saved model checkpoint and evaluate, use the following snippet:
```python
from rom import LDSModel
# other imports

# Load checkpoint stored in 'model/model_checkpoint_xxx' changing <xxx> accordingly
model = LDSModel.load_model("model/model_checkpoint_xxx")

# Use the model however needed
```
