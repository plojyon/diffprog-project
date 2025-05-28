# MABS-SDGD-PINN

Solve the multi-asset Black-Scholes PDE using Physics-Informed Neural Networks (PINNs) and stochastic dimension gradient descent.


## Usage

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Generate synthetic data (or supply your own)
```bash
python generate_data_parameters.py --n_assets 10
python generate_synthetic_data.py
```

3. Train the PINN

```bash
python train.py
```

4. Plot the results

```bash
python report.py
```

Alternatively, use `test.sh` to run the preset experiments.
