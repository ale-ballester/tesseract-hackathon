# PIC-ROM Optimization Tesseract

PIC (Particle-In-Cell) simulation optimization with Fourier actuator control for plasma simulation.

## Installation
Create a conda environment:
```bash
conda create -n hck_tct jax -c conda-forge
```

Build the tesseract:

```bash
conda activate hck_tct
tesseract build .
```

Install runtime dependencies (if running locally):

```bash
conda activate hck_tct
pip install tesseract-core[runtime]
```

## Running the Server

Start the HTTP server:

```bash
conda activate hck_tct
tesseract-runtime serve --host 0.0.0.0 --port 8545
```

The API will be available at:
- **API Docs**: http://localhost:8545/docs
- **ReDoc**: http://localhost:8545/redoc

## Test Cases

### Default parameters

```bash
curl -X POST http://localhost:8545/apply \
  -H "Content-Type: application/json" \
  -d '{"inputs": {}}'
```

### Custom parameters

```bash
curl -X POST http://localhost:8545/apply \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "N_particles": 10000,
      "N_mesh": 200,
      "t1": 10.0,
      "n_steps": 5,
      "lr": 0.05,
      "seed": 42
    }
  }'
```

## Input

All parameters are optional (defaults shown):

<!--     N_particles: int = Field(default=40000, description="Number of particles")
    N_mesh: int = Field(default=400, description="Number of mesh cells")
    t1: float = Field(default=20.0, description="Time at which simulation ends")
    dt: float = Field(default=0.1, description="Timestep")
    boxsize: float = Field(default=50.0, description="Periodic domain [0,boxsize]")
    n0: float = Field(default=1.0, description="Electron number density")
    vb: float = Field(default=3.0, description="Beam velocity")
    vth: float = Field(default=1.0, description="Beam width")
    pos_sample: bool = Field(default=False, description="Whether to sample positions randomly")
    
    # Optimization parameters
    n_steps: int = Field(default=10, description="Number of optimization steps")
    lr: float = Field(default=1e-1, description="Learning rate")
    seed: int = Field(default=0, description="Random seed")
    
    # Initial modes parameters (for FourierActuator initialization)
    mode_n: int = Field(default=1, description="Time mode index")
    mode_m: int = Field(default=1, description="Space mode index")
    mode_A: float = Field(default=1e5, description="Mode amplitude")
    mode_phi_t: float = Field(default=0.0, description="Time phase")
    mode_phi_x: float = Field(default=0.0, description="Space phase") -->

- `N_particles`: 40000 (number of particles)
- `N_mesh`: 400 (mesh cells)
- `t1`: 20.0 (end time)
- `dt`: 0.1 (timestep)
- `boxsize`: 50.0 (periodic domain size)
- `n0`: 1.0 (electron number density)
- `vb`: 3.0 (beam velocity)
- `vth`: 1.0 (beam width)
- `pos_sample`: false (whether to sample positions randomly)
- `n_steps`: 10 (number of optimization steps)
- `lr`: 0.1 (learning rate)
- `seed`: 0 (random seed)
- `mode_n`: 1 (time mode index)
- `mode_m`: 1 (space mode index)
- `mode_A`: 1e5 (mode amplitude)
- `mode_phi_t`: 0.0 (time phase)
- `mode_phi_x`: 0.0 (space phase)

## Output

All outputs are returned in a JSON object.
<!--     train_losses: Array[(None,), Float64] = Field(description="Training losses at each optimization step")
    final_loss: float = Field(description="Final training loss")
    
    # Simulation results after optimization
    positions: Array[(None, None), Float32] = Field(description="Particle positions over time (Nt, Np)")
    velocities: Array[(None, None), Float32] = Field(description="Particle velocities over time (Nt, Np)")
    E_field: Array[(None, None), Float32] = Field(description="Electric field over time (Nt, N_mesh)")
    E_ext: Array[(None, None), Float32] = Field(description="External electric field over time (Nt, N_mesh)")
    rho: Array[(None, None), Float32] = Field(description="Density over time (Nt, N_mesh)")
    momentum: Array[(None, None), Float32] = Field(description="Momentum over time (Nt, N_mesh)")
    energy: Array[(None, None), Float32] = Field(description="Energy over time (Nt, N_mesh)")
    ts: Array[(None,), Float32] = Field(description="Time array") -->
- `train_losses`: Training losses at each optimization step
- `final_loss`: Final training loss
- `positions`: Particle positions over time (Nt, Np)
- `velocities`: Particle velocities over time (Nt, Np)
- `E_field`: Electric field over time (Nt, N_mesh)
- `E_ext`: External electric field over time (Nt, N_mesh)
- `rho`: Density over time (Nt, N_mesh)
- `momentum`: Momentum over time (Nt, N_mesh)
- `energy`: Energy over time (Nt, N_mesh)
- `ts`: Time array

See API docs at http://localhost:8545/docs for full parameter list.
