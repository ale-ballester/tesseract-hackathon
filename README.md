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

### Test 1: Default parameters

```bash
curl -X POST http://localhost:8545/apply \
  -H "Content-Type: application/json" \
  -d '{"inputs": {}}'
```

### Test 2: Small simulation (fast)

```bash
curl -X POST http://localhost:8545/apply \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "N_particles": 1000,
      "N_mesh": 50,
      "t1": 5.0,
      "n_steps": 2
    }
  }'
```

### Test 3: Custom parameters

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

### Test 4: Pretty print output

```bash
curl -X POST http://localhost:8545/apply \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"N_particles": 1000, "N_mesh": 50, "n_steps": 2}}' \
  | python -m json.tool
```

## Input Parameters

All parameters are optional (defaults shown):

- `N_particles`: 40000 (number of particles)
- `N_mesh`: 400 (mesh cells)
- `t1`: 20.0 (end time)
- `dt`: 0.1 (timestep)
- `n_steps`: 10 (optimization steps)
- `lr`: 0.1 (learning rate)
- `seed`: 0 (random seed)

See API docs at http://localhost:8545/docs for full parameter list.
