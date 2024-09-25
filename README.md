# k_cycle

A multigrid kappa-cycle implementation:

C++ implementation of "On the Recursive Structure of Multigrid Cycles" by Or Avnat and Irad Yavneh

Paper URLs: https://doi.org/10.1137/21M1433502, https://arxiv.org/abs/2010.00626

This code implements a multigrid "kappa-cycle" type, which is a family of cycle types which includes the classic "v-cycle", "w-cycle" and "f-cycle" types. The code may run on the CPU or on the GPU using cuda, decided by a command line option. The code also implements a Conjugate Gradients solver with a multigrid cycle as a preconditioner.

## Citation

If you find this code useful, please cite our paper:

```
@article{doi:10.1137/21M1433502,
author = {Avnat, Or and Yavneh, Irad},
title = {On the Recursive Structure of Multigrid Cycles},
journal = {SIAM Journal on Scientific Computing},
volume = {45},
number = {3},
pages = {S103-S126},
year = {2023},
doi = {10.1137/21M1433502}
}
```
## Building 

### Requirements

* A C++ compiler with openmp support
* Cuda, including the cublas and cusparse libraries that comes with cuda

The code does not depend on any other external library.

A CPU only version of the program can also be built without the cuda requirement.

As far as I know, visual studio is the only compiler supported by cuda on windows, so you must use visual studio if you want a GPU capable version of the program on windows.

### Building on Windows

A visual studio solution with 2 project files are provided. You can build a CPU-only version of the program using the file k_cycle_no_cuda.vcxproj and a GPU capable version using the file k_cycle_cuda.vcxproj. The GPU project file currently assumes cuda version 12.6, if you have another version you may need to update this setting or re-create the project file with the appropriated cuda version.

In order to build the CPU-only executable file without the project file, you need to compile and link the following C++ files: main.cpp, Matrix2D.cpp, MG2D.cpp, Level.cpp, ConjugateGradient.cpp, functions.cpp. You also need to specify Stack Reverse Size of 100000000.

In addition, for cuda support, the file cuda_functions.cu must be compiled using the cuda compiler and you must define a preprocessor macro with the name HAVE_CUDA when compiling each of the files (both cpp files and cu files). You also need to link with the library files cublas.lib and cusparse.lib.

### Building on linux

On linux, the code has been tested using the gcc compiler only, but should be able to run using any C++ compiler that suports cuda.

You can build the executable file using the command:

```
nvcc main.cpp MG2D.cpp Matrix2D.cpp Level.cpp ConjugateGradient.cpp functions.cpp cuda_functions.cu -o k_cycle_cuda.out -O3 -Xcompiler -fopenmp -DHAVE_CUDA -lcublas -lcusparse
```
