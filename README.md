# Distributed MST using GHS

This project implements and evaluates distributed Minimum Spanning Tree algorithms, with a focus on the Gallager–Humblet–Spira (GHS) algorithm and related controlled variants.

## Overview

The goal of the project is to study how a Minimum Spanning Tree can be constructed in a distributed environment where nodes communicate with each other and make decisions using only locally available information.

The project includes:

* Implementation of distributed MST algorithms
* GHS-based MST construction
* Controlled variants of GHS
* Test graph datasets
* Experimental results and plots
* Performance analysis using different graph configurations

## Project Structure

```text
.
├── src/                # Source code
├── test_graphs/        # Input graphs used for testing
├── results/            # Experimental results and plots
├── analysis/           # Notebooks/scripts for analysis
├── docs/               # Project documentation/report
└── README.md
```

The exact structure may vary depending on the current version of the repository.

## Technologies Used

* C++
* MPI / Distributed Programming
* Python
* Jupyter Notebook

## Algorithms

The project primarily explores:

* Minimum Spanning Tree
* Gallager–Humblet–Spira (GHS) Algorithm
* Controlled GHS
* Distributed graph processing

## Running the Project

Compile the required C++ source file using a suitable C++ compiler.

Example:

```bash
g++ source.cpp -o program
./program
```

For MPI-based programs:

```bash
mpic++ source.cpp -o program
mpirun -np <number_of_processes> ./program
```

Update the filenames and number of processes according to the implementation being executed.

## Results

The repository contains experimental outputs and plots generated using different graph sizes and configurations. These experiments are used to compare the behaviour and performance of the implemented distributed MST approaches.

## Purpose

This project was developed as part of academic work on Distributed Computing and is intended to demonstrate the design, implementation, and evaluation of distributed graph algorithms.
