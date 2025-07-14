#!/bin/bash

# Set the base directory for the project
BASE_DIR=$(dirname "$(dirname "$(realpath "$0")")")

# Set the Python executable
PYTHON_EXEC=$(which python)

# Set the arguments for the performance test
# Note: Argument names must match those in perf_test.py
# --warmup and --repeats are hardcoded in the Python script and not accepted as arguments.
args=(
    --M 4096
    --K 4096
    --N 4096
    --block-size 32
)

# Run the performance test
$PYTHON_EXEC $BASE_DIR/gemm_test/perf_test.py "${args[@]}"