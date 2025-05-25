# MABS-SDGD-PINN

Solve the multi-asset Black-Scholes PDE using Physics-Informed Neural Networks (PINNs) and stochastic dimension gradient descent.


## Usage

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Generate synthetic data (or supply your own)
```bash
python generate_syntetic_data.py
```

3. Train the PINN

```bash
python train.py
```

4. Evaluate the model

```bash
python evaluate.py
```

## Parameters

### data_parameters.json

- `K`: strike price
- `T`: time to maturity
- `r`: risk-free rate
- `sigmas`: volatilities of the assets
- `rho`: correlation matrix of the assets
- `N_data`: number of data points
- `S0`: initial prices of the assets
- `bias`: bias term

### model_parameters.json

- `epochs`: number of training epochs
- `lr`: learning rate
- `alpha`: regularization parameter
- `path`: path to save the model
- `colloc_min_S`: minimum value of the state space for collocation points
- `colloc_max_S`: maximum value of the state space for collocation points
- `colloc_max_T`: maximum value of the time space for collocation points
- `colloc_count`: number of collocation points
- `hidden_dims`: number of hidden dimensions
