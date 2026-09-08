// controlled_ghs_mpi_3stage_logging_counted.cpp
//
// Compile:
//   mpicxx -std=c++17 -O2 -o controlled_ghs_mpi_3stage_logging_counted controlled_ghs_mpi_3stage_logging_counted.cpp
//
// Run:
//   mpiexec --hostfile hosts -np 5 ./controlled_ghs_mpi_3stage_logging_counted input.txt mst_out.txt
//
// This version is the same as your controlled_ghs_mpi_3stage_logging.cpp but
// with message counting instrumentation added. It counts calls to MPI_Send,
// MPI_Recv and MPI_Bcast (counted as 1 send at root and 1 receive per non-root).
// Each rank still writes to "rank_<r>.log". The root prints aggregated totals
// to stdout at the end.

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <deque>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <memory> // For std::unique_ptr

using namespace std;

// --- Global Logger ---
static std::unique_ptr<std::ofstream> global_log_stream;
static int global_rank = -1; // Holds the rank for the logger

// ---------------- message counting ----------------
static long long messages_sent = 0;
static long long messages_received = 0;

// wrappers that increment counters on successful call
inline int SEND_COUNT(const void* buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm) {
    int rc = MPI_Send(buf, count, type, dest, tag, comm);
    if (rc == MPI_SUCCESS) messages_sent++;
    return rc;
}
inline int RECV_COUNT(void* buf, int count, MPI_Datatype type, int src, int tag, MPI_Comm comm, MPI_Status* st) {
    int rc = MPI_Recv(buf, count, type, src, tag, comm, st);
    if (rc == MPI_SUCCESS) messages_received++;
    return rc;
}
inline int BCAST_COUNT(void* buffer, int count, MPI_Datatype type, int root, MPI_Comm comm) {
    int rc = MPI_Bcast(buffer, count, type, root, comm);
    if (rc == MPI_SUCCESS) {
        int r = -1;
        MPI_Comm_rank(comm, &r);
        if (r == root) messages_sent++;     // root performed one broadcast send (count as 1)
        else messages_received++;           // non-root considered as receiving 1
    }
    return rc;
}
// --------------------------------------------------

struct Edge
{
    int v;
    double w;
};
struct MoE
{
    double w;
    int u;
    int v;
    bool valid;
};

struct UnionFind
{
    int n;
    vector<int> p, r;
    UnionFind(int n_ = 0) : n(n_), p(n_), r(n_, 0)
    {
        p.resize(n);
        for (int i = 0; i < n; i++)
            p[i] = i;
    }
    int find(int x)
    {
        while (p[x] != x)
        {
            p[x] = p[p[x]];
            x = p[x];
        }
        return x;
    }
    bool unite(int a, int b)
    {
        a = find(a);
        b = find(b);
        if (a == b)
            return false;
        if (r[a] < r[b])
            p[a] = b;
        else
        {
            p[b] = a;
            if (r[a] == r[b])
                r[a]++;
        }
        return true;
    }
};

void die(const string& s)
{
    // Log the fatal error to cerr and the log file before aborting
    cerr << "[Rank " << global_rank << "] FATAL: " << s << endl;
    if (global_log_stream && global_log_stream->is_open())
    {
        (*global_log_stream) << "FATAL: " << s << endl;
        global_log_stream->close();
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
}

// ---------- Tree broadcast (root -> all) implemented by parent->children forwarding ----------
void tree_broadcast_ints(const vector<int>& parent, const vector<vector<int>>& children,
    int root, vector<int>& buffer /* in root: data; others: resized to L */)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int L = buffer.size();

    // Use a macro for easy logging
#define LOG (*global_log_stream) << "[Rank " << rank << "] "

    if (rank == root)
    {
        LOG << "BCAST: Root broadcasting " << L << " ints to " << children[rank].size() << " children." << endl;
        for (int c : children[rank])
        {
            LOG << "BCAST: Sending to child " << c << endl;
            SEND_COUNT(buffer.data(), L, MPI_INT, c, 100, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Status st;
        LOG << "BCAST: Waiting to recv from parent " << parent[rank] << endl;
        RECV_COUNT(buffer.data(), L, MPI_INT, parent[rank], 100, MPI_COMM_WORLD, &st);
        LOG << "BCAST: Received. Forwarding to " << children[rank].size() << " children." << endl;
        for (int c : children[rank])
        {
            LOG << "BCAST: Forwarding to child " << c << endl;
            SEND_COUNT(buffer.data(), L, MPI_INT, c, 100, MPI_COMM_WORLD);
        }
    }
#undef LOG
}

// ---------- Tree gather of per-node MoE choices to root ----------
void tree_gather_moes(const vector<int>& parent, const vector<vector<int>>& children,
    int root, const array<double, 3>& my_choice, vector<array<double, 3>>& out_at_root)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#define LOG (*global_log_stream) << "[Rank " << rank << "] "

    vector<array<double, 3>> aggregated;
    // Add this node's own choice
    aggregated.push_back(my_choice);

    LOG << "GATHER: My choice: w=" << my_choice[0] << " u=" << my_choice[1] << " v=" << my_choice[2] << endl;

    // Receive from all children
    for (int c : children[rank])
    {
        int child_count;
        MPI_Status st;
        LOG << "GATHER: Waiting to recv count from child " << c << endl;
        RECV_COUNT(&child_count, 1, MPI_INT, c, 201, MPI_COMM_WORLD, &st);
        LOG << "GATHER: Child " << c << " has " << child_count << " items." << endl;

        if (child_count > 0)
        {
            vector<double> buf(3 * child_count);
            RECV_COUNT(buf.data(), 3 * child_count, MPI_DOUBLE, c, 202, MPI_COMM_WORLD, &st);
            for (int i = 0; i < child_count; i++)
            {
                array<double, 3> t{ buf[3 * i + 0], buf[3 * i + 1], buf[3 * i + 2] };
                aggregated.push_back(t);
            }
            LOG << "GATHER: Received " << child_count << " items from child " << c << endl;
        }
    }

    if (rank == root)
    {
        LOG << "GATHER: Root received. Total items: " << aggregated.size() << endl;
        out_at_root = std::move(aggregated);
        return;
    }
    else
    {
        // Send aggregated list up to parent
        int cnt = aggregated.size();
        LOG << "GATHER: Sending " << cnt << " items up to parent " << parent[rank] << endl;
        SEND_COUNT(&cnt, 1, MPI_INT, parent[rank], 201, MPI_COMM_WORLD);

        if (cnt > 0)
        {
            vector<double> flat(3 * cnt);
            for (int i = 0; i < cnt; i++)
            {
                flat[3 * i + 0] = aggregated[i][0];
                flat[3 * i + 1] = aggregated[i][1];
                flat[3 * i + 2] = aggregated[i][2];
            }
            SEND_COUNT(flat.data(), 3 * cnt, MPI_DOUBLE, parent[rank], 202, MPI_COMM_WORLD);
        }
    }
#undef LOG
}

// ---------- Utility: read graph on rank 0 and scatter per-node adjacency ----------
void read_graph_and_scatter(const string& input_file, int rank, int size,
    int& n_out, vector<Edge>& my_adj,
    vector<int>& parent /*out - for all ranks */, vector<vector<int>>& children)
{
#define LOG (*global_log_stream) << "[Rank " << rank << "] "

    int n;
    vector<vector<Edge>> adj;
    if (rank == 0)
    {
        LOG << "SETUP: Rank 0 reading graph from " << input_file << endl;
        ifstream fin(input_file);
        if (!fin.is_open())
            die("Cannot open input file");
        string line;
        vector<string> lines;
        while (getline(fin, line))
        {
            if (line.find_first_not_of(" \t\r\n") != string::npos)
                lines.push_back(line);
        }
        fin.close();
        if (lines.empty())
            die("Empty input");
        n = stoi(lines[0]);
        LOG << "SETUP: Found n=" << n << " nodes." << endl;
        if (n != size)
        {
            die("Error: number of ranks (-n) = " + to_string(size) + " but file declares n = " + to_string(n));
        }
        adj.assign(n, {});
        for (size_t i = 1; i < lines.size(); ++i)
        {
            string ln = lines[i];
            auto pos = ln.find(':');
            if (pos == string::npos)
                continue;
            int u = stoi(ln.substr(0, pos));
            string rest = ln.substr(pos + 1);
            stringstream ss(rest);
            string tok;
            while (ss >> tok)
            {
                auto comma = tok.find(',');
                if (comma == string::npos)
                    continue;
                int v = stoi(tok.substr(0, comma));
                double w = stod(tok.substr(comma + 1));
                adj[u].push_back({ v, w });
            }
        }
        LOG << "SETUP: Symmetrizing graph..." << endl;
        for (int u = 0; u < n; ++u)
        {
            for (auto& e : adj[u])
            {
                bool found = false;
                for (auto& f : adj[e.v])
                    if (f.v == u)
                    {
                        found = true;
                        break;
                    }
                if (!found)
                    adj[e.v].push_back({ u, e.w });
            }
        }

        LOG << "SETUP: Computing BFS control tree..." << endl;
        parent.assign(n, -1);
        deque<int> q;
        vector<int> seen(n, 0);
        q.push_back(0);
        seen[0] = 1;
        parent[0] = -1;
        while (!q.empty())
        {
            int u = q.front();
            q.pop_front();
            for (auto& e : adj[u])
            {
                int v = e.v;
                if (!seen[v])
                {
                    seen[v] = 1;
                    parent[v] = u;
                    q.push_back(v);
                }
            }
        }
        for (int i = 0; i < n; i++)
            if (!seen[i])
                parent[i] = -1;
        children.assign(n, {});
        for (int i = 0; i < n; i++)
            if (parent[i] >= 0)
                children[parent[i]].push_back(i);

        LOG << "SETUP: Scattering adjacency lists to all ranks..." << endl;
        for (int dest = 0; dest < n; ++dest)
        {
            if (dest == 0)
            {
                my_adj = adj[0];
                continue;
            }
            int m = adj[dest].size();
            SEND_COUNT(&m, 1, MPI_INT, dest, 11, MPI_COMM_WORLD);
            if (m > 0)
            {
                vector<int> vs(m);
                vector<double> ws(m);
                for (int i = 0; i < m; ++i)
                {
                    vs[i] = adj[dest][i].v;
                    ws[i] = adj[dest][i].w;
                }
                SEND_COUNT(vs.data(), m, MPI_INT, dest, 12, MPI_COMM_WORLD);
                SEND_COUNT(ws.data(), m, MPI_DOUBLE, dest, 13, MPI_COMM_WORLD);
            }
        }
    }

    LOG << "SETUP: Broadcasting n=" << n << endl;
    BCAST_COUNT(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    n_out = n;

    if (rank != 0)
    {
        LOG << "SETUP: Receiving my adjacency list..." << endl;
        int m;
        MPI_Status st;
        RECV_COUNT(&m, 1, MPI_INT, 0, 11, MPI_COMM_WORLD, &st);
        if (m > 0)
        {
            vector<int> vs(m);
            vector<double> ws(m);
            RECV_COUNT(vs.data(), m, MPI_INT, 0, 12, MPI_COMM_WORLD, &st);
            RECV_COUNT(ws.data(), m, MPI_DOUBLE, 0, 13, MPI_COMM_WORLD, &st);
            my_adj.resize(m);
            for (int i = 0; i < m; ++i)
            {
                my_adj[i].v = vs[i];
                my_adj[i].w = ws[i];
            }
        }
        else
        {
            my_adj.clear();
        }
        LOG << "SETUP: Received " << m << " neighbors." << endl;
    }

    LOG << "SETUP: Broadcasting BFS tree structure..." << endl;
    if (rank == 0)
    {
        BCAST_COUNT(parent.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
        vector<int> child_counts(n);
        for (int i = 0; i < n; i++)
            child_counts[i] = children[i].size();
        BCAST_COUNT(child_counts.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
        int total = 0;
        for (int c : child_counts)
            total += c;
        vector<int> allkids;
        allkids.reserve(total);
        for (int i = 0; i < n; i++)
            for (int kid : children[i])
                allkids.push_back(kid);
        if (total > 0)
            BCAST_COUNT(allkids.data(), total, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else
    {
        parent.assign(n, -1);
        BCAST_COUNT(parent.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
        vector<int> child_counts(n);
        BCAST_COUNT(child_counts.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
        int total = 0;
        for (int c : child_counts)
            total += c;
        vector<int> allkids(total);
        if (total > 0)
            BCAST_COUNT(allkids.data(), total, MPI_INT, 0, MPI_COMM_WORLD);
        children.assign(n, {});
        int idx = 0;
        for (int i = 0; i < n; i++)
        {
            int c = child_counts[i];
            for (int j = 0; j < c; j++)
            {
                children[i].push_back(allkids[idx++]);
            }
        }
    }
    LOG << "SETUP: BFS tree received. My parent=" << parent[rank] << ", My children=" << children[rank].size() << endl;
#undef LOG
}

// ---------- Relabel fragment IDs to be compact (0..k-1) ----------
void relabel_fragments(int n, vector<int>& frag_ids, int rank)
{
    if (rank == 0)
    {
        (*global_log_stream) << "[Rank 0] Relabeling fragment IDs..." << endl;
    }
    vector<int> uniq = frag_ids;
    sort(uniq.begin(), uniq.end());
    uniq.erase(unique(uniq.begin(), uniq.end()), uniq.end());
    unordered_map<int, int> mapto;
    for (size_t i = 0; i < uniq.size(); ++i)
        mapto[uniq[i]] = (int)i;
    for (int i = 0; i < n; ++i)
        frag_ids[i] = mapto[frag_ids[i]];
}

// ---------- Main algorithm (controlled rounds) ----------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- Logger Initialization ---
    global_rank = rank; // Set global rank for die()
    std::string log_filename = "rank_" + std::to_string(rank) + ".log";
    global_log_stream = std::make_unique<std::ofstream>(log_filename);

    if (!global_log_stream->is_open())
    {
        std::cerr << "[Rank " << rank << "] FATAL: Could not open log file: " << log_filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#define LOG (*global_log_stream) << "[Rank " << rank << "] "

    LOG << "Process started. Total size=" << size << endl;

    if (argc < 3)
    {
        if (rank == 0)
        {
            cerr << "Usage: mpiexec -n N ./controlled_ghs_mpi_3stage input.txt output_mst.txt\n";
            LOG << "FATAL: Insufficient arguments." << endl;
        }
        MPI_Finalize();
        return 1;
    }
    string input_file = argv[1];
    string output_file = argv[2];
    LOG << "Input file: " << input_file << ", Output file: " << output_file << endl;

    // ===================================================================
    // STAGE I: SETUP
    // ===================================================================
    LOG << "--- STAGE I: SETUP ---" << endl;
    int n;
    vector<Edge> my_adj;
    vector<int> parent;
    vector<vector<int>> children;
    read_graph_and_scatter(input_file, rank, size, n, my_adj, parent, children);
    LOG << "SETUP: Complete." << endl;

    int root = 0;
    vector<int> frag_ids(n); // Current fragment ID for each node
    if (rank == root)
    {
        for (int i = 0; i < n; i++)
            frag_ids[i] = i;
    }
    else
    {
        frag_ids.assign(n, -1);
    }

    int my_parent = parent[rank];
    vector<int> my_children = children[rank];

    set<tuple<int, int, double>> mst_edges; // Final MST edges (only at root)

    LOG << "Syncing all ranks before timer start..." << endl;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    LOG << "Timer started." << endl;

    // ===================================================================
    // STAGE II: CONTROLLED-GHS
    // ===================================================================

    int num_base_phases = (int)ceil(log2(max(2.0, sqrt(n))));
    LOG << "--- STAGE II: CONTROLLED-GHS ---" << endl;
    LOG << "Stage II will run for " << num_base_phases << " phases." << endl;

    for (int phase = 0; phase < num_base_phases; ++phase)
    {
        LOG << "--- Stage II, Phase " << phase << " ---" << endl;
        // 1) Broadcast current fragment ids from root
        if (rank != root)
            frag_ids.assign(n, -1);
        tree_broadcast_ints(parent, children, root, frag_ids);
        LOG << "Phase " << phase << ": Received fragment map. My frag_id=" << frag_ids[rank] << endl;

        // 2) Each node computes local MOE
        array<double, 3> my_choice;
        my_choice[0] = -1.0;
        my_choice[1] = (double)rank;
        my_choice[2] = -1.0;
        int u = rank;
        for (auto& e : my_adj)
        {
            int v = e.v;
            if (frag_ids[v] != frag_ids[u])
            {
                if (my_choice[0] < 0 || e.w < my_choice[0] ||
                    (e.w == my_choice[0] && make_pair(min(u, v), max(u, v)) < make_pair((int)min((int)my_choice[1], (int)my_choice[2]), (int)max((int)my_choice[1], (int)my_choice[2]))))
                {
                    my_choice[0] = e.w;
                    my_choice[1] = (double)u;
                    my_choice[2] = (double)v;
                }
            }
        }
        LOG << "Phase " << phase << ": My local MOE: w=" << my_choice[0] << " u=" << my_choice[1] << " v=" << my_choice[2] << endl;

        // 3) Tree-convergecast of all choices to root
        vector<array<double, 3>> all_choices;
        tree_gather_moes(parent, children, root, my_choice, all_choices);

        // 4) Root processes choices using CONTROLLED logic
        vector<int> new_frag_ids(n);
        int distinct_frags_stage2 = 0;
        if (rank == root)
        {
            LOG << "Phase " << phase << ": Root processing " << all_choices.size() << " choices (Controlled logic)..." << endl;
            vector<int> node_frag = frag_ids;

            unordered_map<int, array<double, 3>> frag_best;
            for (auto& tr : all_choices)
            {
                double w = tr[0];
                int uu = (int)round(tr[1]);
                int vv = (int)round(tr[2]);
                if (w < 0)
                    continue;
                int f = node_frag[uu];
                auto it = frag_best.find(f);
                if (it == frag_best.end())
                {
                    frag_best[f] = tr;
                }
                else
                {
                    double w0 = it->second[0];
                    int u0 = (int)round(it->second[1]);
                    int v0 = (int)round(it->second[2]);
                    if (w < w0 || (w == w0 && make_pair(min(uu, vv), max(uu, vv)) < make_pair(min(u0, v0), max(u0, v0))))
                    {
                        frag_best[f] = tr;
                    }
                }
            }
            map<int, vector<pair<int, double>>> frag_graph;
            map<pair<int, int>, array<double, 3>> frag_edge_map;
            for (auto& pair : frag_best)
            {
                int f1 = pair.first;
                double w = pair.second[0];
                int u = (int)round(pair.second[1]);
                int v = (int)round(pair.second[2]);
                int f2 = node_frag[v];
                frag_graph[f1].push_back({ f2, w });
                frag_edge_map[{f1, f2}] = pair.second;
            }
            map<int, int> matching;
            set<int> matched_frags;
            vector<tuple<int, int, double>> edges_to_add;
            for (auto const& [f1, neighbors] : frag_graph)
            {
                if (matched_frags.count(f1))
                    continue;
                int best_f2 = -1;
                double min_w = -1.0;
                for (auto const& [f2, w] : neighbors)
                {
                    if (matched_frags.count(f2))
                        continue;
                    auto it = frag_best.find(f2);
                    if (it != frag_best.end())
                    {
                        int f2_target = node_frag[(int)round(it->second[2])];
                        if (f2_target == f1)
                        {
                            if (best_f2 == -1 || w < min_w)
                            {
                                min_w = w;
                                best_f2 = f2;
                            }
                        }
                    }
                }
                if (best_f2 != -1)
                {
                    matching[f1] = best_f2;
                    matching[best_f2] = f1;
                    matched_frags.insert(f1);
                    matched_frags.insert(best_f2);
                    auto edge = frag_edge_map[{f1, best_f2}];
                    edges_to_add.emplace_back(min((int)round(edge[1]), (int)round(edge[2])), max((int)round(edge[1]), (int)round(edge[2])), edge[0]);
                }
            }
            for (auto const& [f, moe] : frag_best)
            {
                if (matched_frags.count(f) == 0)
                {
                    edges_to_add.emplace_back(min((int)round(moe[1]), (int)round(moe[2])), max((int)round(moe[1]), (int)round(moe[2])), moe[0]);
                }
            }
            UnionFind uf(n);
            unordered_map<int, int> frag_to_rep;
            for (int node = 0; node < n; ++node)
            {
                int f = node_frag[node];
                if (frag_to_rep.find(f) == frag_to_rep.end())
                    frag_to_rep[f] = node;
            }
            for (auto& edge : edges_to_add)
            {
                int u, v;
                double w;
                tie(u, v, w) = edge;
                int rep_u = frag_to_rep[node_frag[u]];
                int rep_v = frag_to_rep[node_frag[v]];
                if (uf.unite(rep_u, rep_v))
                {
                    mst_edges.insert(edge);
                }
            }
            vector<int> temp_new(n);
            for (int node = 0; node < n; ++node)
            {
                int oldf = node_frag[node];
                int repnode = frag_to_rep[oldf];
                temp_new[node] = uf.find(repnode);
            }
            new_frag_ids = temp_new;
            relabel_fragments(n, new_frag_ids, rank);

            unordered_set<int> s(new_frag_ids.begin(), new_frag_ids.end());
            distinct_frags_stage2 = s.size();
            LOG << "Phase " << phase << ": Root processing complete. New distinct fragments: " << distinct_frags_stage2 << endl;
        }

        // 5) Broadcast new fragment ids
        tree_broadcast_ints(parent, children, root, new_frag_ids);
        frag_ids = new_frag_ids; // All nodes update
        LOG << "Phase " << phase << ": Received new fragment map. My new frag_id=" << frag_ids[rank] << endl;

        // 6) Check for early termination
        MPI_Bcast(&distinct_frags_stage2, 1, MPI_INT, root, MPI_COMM_WORLD);
        LOG << "Phase " << phase << ": distinct_frags_stage2 broadcast done." << endl;
        if (distinct_frags_stage2 <= 1)
        {
            LOG << "Phase " << phase << ": Termination condition met (<= 1 fragment). Breaking Stage II." << endl;
            break;
        }
    }
    LOG << "--- STAGE II COMPLETE ---" << endl;

    // ===================================================================
    // STAGE III: MERGING THE REMAINING FRAGMENTS
    // ===================================================================
    LOG << "--- STAGE III: FINAL MERGE ---" << endl;

    int max_rounds_stage3 = (int)ceil(log2(max(2, n))) + 10;
    for (int rnd = 0; rnd < max_rounds_stage3; ++rnd)
    {
        LOG << "--- Stage III, Round " << rnd << " ---" << endl;
        // 1) Broadcast fragment IDs
        if (rnd > 0)
        {
            if (rank != root)
                frag_ids.assign(n, -1);
            tree_broadcast_ints(parent, children, root, frag_ids);
        }
        else
        {
            LOG << "Round " << rnd << ": Using fragment map from Stage II." << endl;
        }
        LOG << "Round " << rnd << ": My frag_id=" << frag_ids[rank] << endl;

        // 2) Each node computes local MOE
        array<double, 3> my_choice;
        my_choice[0] = -1.0;
        my_choice[1] = (double)rank;
        my_choice[2] = -1.0;
        int u = rank;
        for (auto& e : my_adj)
        {
            int v = e.v;
            if (frag_ids[v] != frag_ids[u])
            {
                if (my_choice[0] < 0 || e.w < my_choice[0] ||
                    (e.w == my_choice[0] && make_pair(min(u, v), max(u, v)) < make_pair((int)min((int)my_choice[1], (int)my_choice[2]), (int)max((int)my_choice[1], (int)my_choice[2]))))
                {
                    my_choice[0] = e.w;
                    my_choice[1] = (double)u;
                    my_choice[2] = (double)v;
                }
            }
        }
        LOG << "Round " << rnd << ": My local MOE: w=" << my_choice[0] << " u=" << my_choice[1] << " v=" << my_choice[2] << endl;

        // 3) Tree-convergecast of all choices to root
        vector<array<double, 3>> all_choices;
        tree_gather_moes(parent, children, root, my_choice, all_choices);

        // 4) Root processes all choices (Greedy Logic)
        vector<int> new_frag_ids(n);
        if (rank == root)
        {
            LOG << "Round " << rnd << ": Root processing " << all_choices.size() << " choices (Greedy logic)..." << endl;
            vector<int> node_frag = frag_ids;
            UnionFind uf(n);
            unordered_map<int, array<double, 3>> frag_best;
            for (auto& tr : all_choices)
            {
                double w = tr[0];
                int uu = (int)round(tr[1]);
                int vv = (int)round(tr[2]);
                if (w < 0)
                    continue;
                int f = node_frag[uu];
                auto it = frag_best.find(f);
                if (it == frag_best.end())
                {
                    frag_best[f] = tr;
                }
                else
                {
                    double w0 = it->second[0];
                    int u0 = (int)round(it->second[1]);
                    int v0 = (int)round(it->second[2]);
                    if (w < w0 || (w == w0 && make_pair(min(uu, vv), max(uu, vv)) < make_pair(min(u0, v0), max(u0, v0))))
                    {
                        frag_best[f] = tr;
                    }
                }
            }
            unordered_map<int, int> frag_to_rep;
            for (int node = 0; node < n; ++node)
            {
                int f = node_frag[node];
                if (frag_to_rep.find(f) == frag_to_rep.end())
                    frag_to_rep[f] = node;
            }
            vector<tuple<int, int, double>> added;
            vector<int> frag_keys;
            for (auto& p : frag_best)
                frag_keys.push_back(p.first);
            sort(frag_keys.begin(), frag_keys.end());
            for (int f : frag_keys)
            {
                auto tr = frag_best[f];
                double w = tr[0];
                int uu = (int)round(tr[1]);
                int vv = (int)round(tr[2]);
                int rep_u = frag_to_rep[node_frag[uu]];
                int rep_v = frag_to_rep[node_frag[vv]];
                if (uf.find(rep_u) != uf.find(rep_v))
                {
                    bool merged = uf.unite(rep_u, rep_v);
                    if (merged)
                    {
                        int a = min(uu, vv), b = max(uu, vv);
                        added.emplace_back(a, b, w);
                    }
                }
            }
            vector<int> temp_new(n);
            for (int node = 0; node < n; ++node)
            {
                int oldf = node_frag[node];
                int repnode = frag_to_rep[oldf];
                temp_new[node] = uf.find(repnode);
            }
            new_frag_ids = temp_new;
            relabel_fragments(n, new_frag_ids, rank);
            for (auto& t : added)
            {
                mst_edges.insert(t);
            }
            LOG << "Round " << rnd << ": Root processing complete. " << added.size() << " edges added to MST." << endl;
        }

        // 5) Root tree-broadcasts new_frag_ids
        tree_broadcast_ints(parent, children, root, new_frag_ids);
        frag_ids = new_frag_ids; // All nodes update
        LOG << "Round " << rnd << ": Received new fragment map. My new frag_id=" << frag_ids[rank] << endl;

        // 7) Termination check
        int distinct = 0;
        if (rank == root)
        {
            unordered_set<int> s(new_frag_ids.begin(), new_frag_ids.end());
            distinct = s.size();
        }
        MPI_Bcast(&distinct, 1, MPI_INT, root, MPI_COMM_WORLD);
        LOG << "Round " << rnd << ": Termination check. Distinct fragments = " << distinct << endl;
        if (distinct <= 1)
        {
            LOG << "Round " << rnd << ": Termination condition met. Breaking Stage III." << endl;
            break;
        }
    } // end stage 3 rounds
    LOG << "--- STAGE III COMPLETE ---" << endl;

    LOG << "Syncing all ranks before timer stop..." << endl;
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double total_time = t1 - t0;
    LOG << "Timer stopped." << endl;
    LOG << "=== TIME TO TERMINATION: " << fixed << setprecision(6) << total_time << " seconds ===" << endl;

    // ===================================================================
    // FINALIZATION: Root 0 writes the output
    // ===================================================================
    if (rank == root)
    {
        LOG << "--- FINALIZATION ---" << endl;
        vector<vector<double>> adjmat(n, vector<double>(n, 0.0));
        for (auto& t : mst_edges)
        {
            int a, b;
            double w;
            tie(a, b, w) = t;
            adjmat[a][b] = w;
            adjmat[b][a] = w;
        }
        int edges_count = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (adjmat[i][j] > 0)
                    edges_count++;

        // Also print final summary to stdout
        cout << "[Root] MST edge count: " << edges_count << "\n";
        cout << "[Root] Total merging time (seconds): " << fixed << setprecision(6) << total_time << "\n";
        LOG << "MST edge count: " << edges_count << endl;
        LOG << "Total merging time (seconds): " << fixed << setprecision(6) << total_time << endl;

        ofstream fout(output_file);
        if (!fout.is_open())
            die("Cannot open output file");

        LOG << "Writing MST adjacency matrix to " << output_file << endl;
        fout << n << "\n";
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (j)
                    fout << " ";
                if (adjmat[i][j] > 0)
                    fout << fixed << setprecision(6) << adjmat[i][j];
                else
                    fout << "0";
            }
            fout << "\n";
        }
        fout.close();
        cout << "[Root] MST adjacency matrix written to " << output_file << "\n";
        LOG << "--- ALL DONE ---" << endl;
    }

    // Aggregate message counts (sum across ranks) and print on root
    long long total_sent = 0;
    long long total_recv = 0;
    MPI_Reduce(&messages_sent, &total_sent, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&messages_received, &total_recv, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    LOG << "Messages sent (local): " << messages_sent << ", received (local): " << messages_received << endl;
    if (rank == root)
    {
        cout << "[Root] Total messages sent: " << total_sent << "\n";
        cout << "[Root] Total messages received: " << total_recv << "\n";
        LOG << "[Root] Total messages sent: " << total_sent << ", total messages received: " << total_recv << endl;
    }

    // Close the log file stream
    global_log_stream->close();

    MPI_Finalize();
    return 0;
}
