# pytorch-gemm-test

This project implements a customizable General Matrix Multiply (GEMM) operation using CUDA and PyTorch's C++ extension interface. The goal is to provide a framework for performance testing of GEMM operations while allowing users to specify the number of threads and blocks for the CUDA kernel.

## Project Structure

- **csrc/**: Contains the C++ source files for the CUDA kernel and bindings.
  - **gemm_kernel.cu**: Implements the CUDA kernel for GEMM, allowing customizable thread block sizes.
  - **bindings.cpp**: Provides C++ bindings to expose the GEMM operation to Python.

- **gemm_test/**: Contains the performance testing logic.
  - **__init__.py**: Marks the directory as a Python package.
  - **perf_test.py**: Sets up the environment for GEMM operations and measures performance.

- **scripts/**: Contains scripts for running tests.
  - **run_perf_test.sh**: Shell script to execute the performance test.

- **setup.py**: Configuration file for building the C++ extension.

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd pytorch-gemm-test
   ```

2. **Install dependencies**:
   Ensure you have PyTorch installed with CUDA support. You can install it via pip:
   ```
   pip install torch torchvision torchaudio
   ```

3. **Build the C++ extension**:
   Run the following command to build the extension:
   ```
   python setup.py install
   ```

## Usage

To run the performance tests, execute the following script:
```
bash scripts/run_perf_test.sh
```

This will set up the environment and run the performance tests defined in `gemm_test/perf_test.py`.

## Performance Testing Methodology

The performance tests measure the impact of overlapping computation and communication during GEMM operations. The tests allow for customization of thread block sizes, enabling users to explore the performance characteristics of different configurations.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.