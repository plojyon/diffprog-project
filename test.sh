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
    "epochs": 1000000000,
    "lr": 1e-3,
    "alpha": 0.01,
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
    
    echo "Running $model_dir ..."
    python generate_data_parameters.py --n_assets "$n_assets" --output "$model_dir/data_parameters.json"
    python generate_synthetic_data.py --data_parameters "$model_dir/data_parameters.json" --output "$model_dir/data.json"
    python train.py --model_parameters "$model_dir/model_parameters.json" --data "$model_dir/data.json" --time_limit_minutes 60
    echo "----------------------------------------"
}

experiments=(
    "3:1"
    "3:3"
    "30:1"
    "30:10"
    "30:30"
    "100:1"
    "100:5"
    "100:100"
)

for entry in "${experiments[@]}"; do
    IFS=':' read -r n_assets sdgd_dim_count <<< "$entry"
    model_dir="models/$n_assets-$sdgd_dim_count"
    create_model_params "$model_dir" "$n_assets" "$sdgd_dim_count"
    run_experiment "$model_dir" "$n_assets"
done
