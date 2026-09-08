# Distributed BFS + MST (MPI) — Complete README

## 📌 Overview

This project implements a **fully distributed Minimum Spanning Tree (MST)** algorithm using **MPI**. It follows:

- BFS-based spanning tree construction (rank 0 only, no MPI)
- Algorithm 6 (Borůvka-style distributed MST)
- Safe MPI send/receive (deadlock-safe)
- JSON graph input format
- Automatic BFS → `topology.json` → distributed MST

The entire system runs in **two phases**:

1. **Rank 0** builds a BFS communication tree and saves `topology.json`
2. All ranks load the topology and execute the MST using MPI

---

## 🧩 Features

✔ Automatic BFS topology generation  
✔ Fully distributed MST algorithm  
✔ Safe, timeout-protected MPI communication  
✔ CSV logging for experimental analysis  
✔ Works on macOS (Homebrew) and Linux  
✔ Handles dense & sparse graphs  
✔ Prints final MST edges  

---

# 1. System Requirements

### Operating Systems
- macOS (Intel + M-Series)
- Ubuntu / Debian / Fedora Linux
- HPC clusters with OpenMPI

### Software
- OpenMPI
- C++17 compiler
- Homebrew (macOS only)

---

# 2. Installation Instructions

## macOS Setup

### Step 1: Install Homebrew
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Verify:
```
brew --version
```

### Step 2: Install MPI
```
brew install open-mpi
```

Verify:
```
mpic++ --version
mpirun --version
```

---

## Ubuntu / Linux Setup

Install OpenMPI:
```
sudo apt update
sudo apt install openmpi-bin openmpi-common libopenmpi-dev -y
```

Verify:
```
mpic++ --version
mpirun --version
```

---

# 3. Project Structure

```
project/
│── Src_Prjt-cs24mtech12002-2.cpp   # Combined BFS + MST implementation
│── graph.json                       # Graph input file
│── topology.json                    # Auto-generated BFS tree
│── topology_results.csv             # CSV experiment logs
│── README.md                        # This file
```

---

# 4. Graph Input Format (`graph.json`)

Example:
```json
{
  "nodes": 8,
  "topology": "complete",
  "edges": [
    {"u": 0, "v": 1, "w": 4},
    {"u": 1, "v": 2, "w": 8},
    {"u": 2, "v": 3, "w": 7},
    {"u": 3, "v": 4, "w": 9},
    {"u": 4, "v": 5, "w": 10},
    {"u": 5, "v": 6, "w": 2},
    {"u": 6, "v": 7, "w": 1},
    {"u": 7, "v": 0, "w": 8}
  ]
}
```

**NOTE:**  
`nodes` must equal the number of MPI processes.

---

# 5. Compiling the Code

## macOS (Homebrew OpenMPI)
```
mpic++ -O3 -std=c++17 -I/opt/homebrew/include -o mst Src_Prjt-cs24mtech12002-2.cpp
```

## Linux
```
mpic++ -O3 -std=c++17 -o mst Src_Prjt-cs24mtech12002-2.cpp
```

---

# 6. Running the Program

Use:
```
mpirun -np <numProcesses> --oversubscribe ./mst graph.json topology.json
```

Example:
```
mpirun -np 10 --oversubscribe ./mst graph.json topology.json
```

Execution flow:
1. Rank 0 runs BFS → saves `topology.json`  
2. All ranks load graph and topology  
3. All ranks run MST  
4. Rank 0 prints:
   - MST edges
   - Total time
   - Total messages
   - Phase count
   - Fragment states
   - Density  
   - Appends a row to `topology_results.csv`

---

# 7. Output Files

### `topology.json`
Generated automatically.

### `topology_results.csv`
Each run appends:
```
n, topologyName, totalTime, totalMessages, density, phases
```

---

# 8. MST Output

At the end:
```
===== FINAL MST EDGES =====
0 -- 3  [w=6]
3 -- 4  [w=7]
4 -- 7  [w=2]
...
Total MST edges = 7 (expected 7)
```

---

# 9. Troubleshooting

### Deadlock detected
```
[DEADLOCK] Rank X waiting for Y during <step>
```

### MPI not found (macOS M1/M2)
```
brew reinstall open-mpi
brew link --overwrite open-mpi
```

### No MST edges printed
Graph may be disconnected.

---

# 10. Cleaning Build Files
```
rm -f mst topology.json topology_results.csv
```

---

# End of README
