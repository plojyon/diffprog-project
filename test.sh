#!/bin/bash

source venv/bin/activate

# Function to create model parameters JSON
create_model_params() {
    local model_dir=$1
    local n_assets=$2
    local sdgd_dim_count=$3
    
    mkdir -p "$model_dir"
    cat > "$model_dir/model_parameters.json" << EOF
{
    "epochs": 50000,
    "lr": 1e-3,
    "alpha": 0.1,
    "path": "$model_dir",
    "colloc_count": 100,
    "n_assets": $n_assets,
    "hidden_dims": [16, 16],
    "sdgd_dim_count": $sdgd_dim_count
}
EOF
}

run_experiment() {
    local model_dir=$1
    local n_assets=$2
    
    echo "Running $model_dir ($n_assets assets) ..."
    python generate_data_parameters.py --n_assets "$n_assets" --output "$model_dir/data_parameters.json"
    python generate_synthetic_data.py --data_parameters "$model_dir/data_parameters.json" --output "$model_dir/data.json"
    python train.py --model_parameters "$model_dir/model_parameters.json" --data "$model_dir/data.json"

    echo "Completed $model_dir"
    echo "----------------------------------------"
}

declare -A experiments=(
    ["models/3-3"]="3:3"
    ["models/30-30"]="30:30"
    ["models/30-10"]="30:10"
    ["models/30-1"]="30:1"
)

for model_dir in "${!experiments[@]}"; do
    IFS=':' read -r n_assets sdgd_dim_count <<< "${experiments[$model_dir]}"
    create_model_params "$model_dir" "$n_assets" "$sdgd_dim_count"
    run_experiment "$model_dir" "$n_assets"
done
