#!/bin/bash

# Define the gpu to run the script on
GPUID=0

# Define values for each hyperparameter
radius_VALUES=(0.03 0.05)
alpha_VALUES=(0.2 0.4)
rho_VALUES=(0.25 0.45)
eps_VALUES=(0.06 0.08)
zeta_VALUES=(0.375 0.625)

# Create the gpubashresults directory if it doesn't exist
RESULTS_DIR="gpu${GPUID}bashresults"
mkdir -p $RESULTS_DIR

# Function to run the pipeline with a given hyperparameter name and value
run_experiment() {
    param_name=$1
    param_value=$2

    echo "Running pipeline with $param_name=$param_value"

    # Run the training script
    python train.py --gpus $GPUID --ckpt_path "./LOGS${GPUID}/CKPTS/" --$param_name $param_value

    # Run the evaluation script
    python evaluations.py --device $GPUID --model_path "LOGS${GPUID}/CKPTS/STEPS" --dest "log${GPUID}"

    # Copy the evaluation results to gpubashresults with a unique name
    EVAL_RESULTS="log${GPUID}/evaluation_results.csv"
    if [ -f "$EVAL_RESULTS" ]; then
        cp $EVAL_RESULTS "$RESULTS_DIR/${param_name}-${param_value}-evaluation_results.csv"
        echo "Results for $param_name=$param_value saved to $RESULTS_DIR/${param_name}-${param_value}-evaluation_results.csv"
    else
        echo "Error: $EVAL_RESULTS not found!"
    fi

    # Clean up the log and LOGS directories
    echo "Cleaning up log${GPUID}/ and LOGS${GPUID}/ directories..."
    rm -rf "log${GPUID}/" "LOGS${GPUID}/"
    echo "Cleanup complete."
}

# Run experiments for each hyperparameter independently
for radius in "${radius_VALUES[@]}"; do
    run_experiment "radius-gw" $radius
done

for alpha in "${alpha_VALUES[@]}"; do
    run_experiment "alpha-train" $alpha
done

for rho in "${rho_VALUES[@]}"; do
    run_experiment "rho" $rho
done

for eps in "${eps_VALUES[@]}"; do
    run_experiment "eps-train" $eps
done

for zeta in "${zeta_VALUES[@]}"; do
    run_experiment "zeta" $zeta
done

echo "All experiments completed."