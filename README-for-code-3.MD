# Distributed MST using GHS Algorithm

Implementation of the Gallager-Humblet-Spira algorithm for finding Minimum Spanning Trees using MPI.

---

## Installation

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install mpich g++
```

**macOS:**
```bash
brew install mpich
```

**Compile:**
```bash
mpic++ -o ghs_mst Src_Prjt-cs24mtech12002-3.cpp
```

---

## Input Format

Adjacency matrix file format:

```
N
w[0][0] w[0][1] ... w[0][N-1]
w[1][0] w[1][1] ... w[1][N-1]
...
w[N-1][0] ... w[N-1][N-1]
```

- **N**: Number of nodes
- **w[i][j]**: Edge weight (use `100000` for no edge)
- Matrix must be symmetric
- Diagonal values = `100000`

**Example (4 nodes):**
```bash
cat > graph.txt << EOF
4
100000 5 100000 100000
5 100000 6 100000
100000 6 100000 7
100000 100000 7 100000
EOF
```

---

## Running

**Basic usage:**
```bash
mpirun -np 4 --oversubscribe ./ghs_mst graph.txt
```

**With timeout (recommended):**
```bash
timeout 30 mpirun -np 10 --oversubscribe ./ghs_mst graph.txt
```

**Save output:**
```bash
mpirun -np 8 --oversubscribe ./ghs_mst graph.txt > output.txt 2>&1
```

**Important**: Number of processes (`-np N`) must match number of nodes in graph file.

---

## Output Format

**Performance Metrics:**
```
METRICS,<node_id>,<time>,<msgs_sent>,<msgs_received>,<edges>
```

**MST Edges:**
```
MST,<node1>,<node2>,<weight>
```

**Example:**
```
METRICS,0,0.001234,25,22,3
METRICS,1,0.001256,23,24,2
MST,0,1,5
MST,1,2,6
MST,2,3,7
```

---

## Testing

**Quick test:**
```bash
# 1. Create test graph (4 nodes)
cat > test.txt << EOF
4
100000 5 100000 100000
5 100000 6 100000
100000 6 100000 7
100000 100000 7 100000
EOF

# 2. Run
mpirun -np 4 --oversubscribe ./ghs_mst test.txt

# 3. Check MST edges
grep "MST" output.txt
```


**Version**: 1.0 | **Requirements**: MPICH 4.0+, GCC 9+