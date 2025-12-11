#!/bin/bash
# Ensure a mode is provided
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 [--denoising|--inpainting|--deconvolution]"
  exit 1
fi

MODE="$1"  # This will be passed to the Python script
MAX_JOBS=2

#L_VALUES=(10 1)
L_VALUES=(10)
R_VALUES=(10)
#R_VALUES=(10 0 1)
SEEDS=(0)
ZERO_INIT_FLAGS=("")
#ZERO_INIT_FLAGS=("--zero_init")
#ZERO_INIT_FLAGS=("" "--zero_init")
RAND_WEIGHTS_FLAGS=("")
#RAND_WEIGHTS_FLAGS=("" "--rand_weights")
LEARNING_RATES=(0.001 0.0001 0.00001 0.000001)
BATCH_SIZE=(1 4 8 16 32)
mkdir -p logs

job_count=0

for L in "${L_VALUES[@]}"; do
  for R in "${R_VALUES[@]}"; do
    for ZF in "${ZERO_INIT_FLAGS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        for RF in "${RAND_WEIGHTS_FLAGS[@]}"; do
          for LR in "${LEARNING_RATES[@]}"; do
            for B in "${BATCH_SIZE[@]}"; do

              echo "Running: python PGDDenoisingPrior-learn-div2k_learning_tests.py $MODE --L_steps $L --R_restarts $R --batch_size $B --learning_rate "$LR" $ZF $RF"
              
              # Construct log file name
              log_name="L${L}_R${R}"
              [[ $ZF == "--zero_init" ]] && log_name="${log_name}_ZI"
              [[ $RF == "--rand_weights" ]] && log_name="${log_name}_RW"
    
              # Run the command
              python PGDDenoisingPrior-learn-div2k_learning_tests.py $MODE --L_steps "$L" --R_restarts "$R" --batch_size "$B" --learning_rate "$LR" $ZF $RF \
                > "logs/PGDDenoisingPrior-learn-div2k_learning_tests_${log_name}.out" 2>&1 &
    
              ((job_count++))
              if (( job_count % MAX_JOBS == 0 )); then
                wait
              fi
            done
          done
        done
      done
    done
  done
done

wait