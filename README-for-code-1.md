Distributed MST using GHS Algorithm with MPI

PREREQUISITES:
- OpenMPI or MPICH
- C++ compiler with C++17 support
- SSH setup with passwordless access (for multi-system)

SINGLE SYSTEM:

1. Compile
   mpicxx -std=c++17 -O2 -o controlled_ghs_mpi_3stage_logging Src_Prjt-cs24mtech12002-1.cpp

2. Run (n must match number of vertices in input.txt)
   mpiexec -np 5 ./controlled_ghs_mpi_3stage_logging input.txt mst_out.txt

3. View results
   cat mst_out.txt


MULTIPLE SYSTEMS:

Setup:
1. Edit hosts.txt with your host IPs and slot distribution
2. Edit deploy_and_run.sh with your host details and usernames

Run:
   chmod +x deploy_and_run.sh
   ./deploy_and_run.sh

The script automatically deploys code, compiles, runs across hosts, and fetches results.


INPUT FORMAT (input.txt):
First line = number of vertices (n)
Each line = vertex: neighbor,weight pairs

Example:
5
0: 1,1.0 4,2.0 2,3.0
1: 0,1.0 2,1.0
2: 1,1.0 0,3.0 3,1.0
3: 2,1.0 4,1.0
4: 3,1.0 0,2.0


OUTPUT FILES:
- mst_out.txt (MST edges and total weight)
- rank_*.log (per-process execution logs)
- results.csv (performance metrics)
