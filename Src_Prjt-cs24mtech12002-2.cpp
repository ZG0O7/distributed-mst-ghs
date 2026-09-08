// ============================================================================
// Distributed MST with DISC-Lite + Multi-Merge + CONNECT-only message counting
// Fixed: avoid MPI_Allgather buffer aliasing and small robustness tweaks
// ============================================================================

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <string>
#include <nlohmann/json.hpp>
#include <chrono>
#include <climits>
#include <algorithm>
#include <cstdlib>   // for rand, srand
#include <ctime>     // for time

using json = nlohmann::json;
using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// GLOBAL STATE
// ============================================================================
int RANK = -1, SIZE = -1;
bool isRoot = false;

bool COUNTING_ENABLED = false;
long long messageCount = 0;

int phaseCounter = 0;

struct Edge { int u, v, w; };

std::vector<Edge> edges;
std::vector<std::vector<int>> adj;

std::vector<int> fragRoot;
std::vector<int> fragRank;
std::vector<Edge> mst_edges;

std::vector<int> parentT;
std::vector<std::vector<int>> childrenT;

constexpr double DEADLOCK_TIMEOUT = 5.0;

// ============================================================================
// MESSAGE TYPES
// ============================================================================
enum MsgType {
    MSG_RANK_REQUEST,
    MSG_PROCEED,
    MSG_CONNECT,
    MSG_DONE
};

struct Msg {
    int type;
    int a, b, c, d;
};

// ============================================================================
// SAFE SEND / SAFE RECV (COUNT ONLY MSG_CONNECT)
// ============================================================================
void safe_send(int dest, const Msg& m, const char* ctx)
{
    int rc = MPI_Send((void*)&m, sizeof(Msg), MPI_BYTE, dest, m.type, MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS) {
        fprintf(stderr, "[SEND ERR] Rank %d -> %d in %s\n", RANK, dest, ctx);
        MPI_Abort(MPI_COMM_WORLD, 97);
    }

    // Count only MST messages (fragment proposals + merges)
    if (COUNTING_ENABLED && m.type == MSG_CONNECT)
        messageCount++;
}

void safe_recv(int src, Msg& m, int tag, const char* ctx)
{
    MPI_Status status;
    double start = MPI_Wtime();

    while (true) {
        int flag = 0;
        MPI_Iprobe(src, tag, MPI_COMM_WORLD, &flag, &status);

        if (flag) {
            int rc = MPI_Recv(&m, sizeof(Msg), MPI_BYTE, src, tag, MPI_COMM_WORLD, &status);
            if (rc != MPI_SUCCESS) {
                fprintf(stderr, "[RECV ERR] Rank %d <- %d in %s\n", RANK, src, ctx);
                MPI_Abort(MPI_COMM_WORLD, 98);
            }

            // Count only MST messages (CONNECT)
            if (COUNTING_ENABLED && m.type == MSG_CONNECT)
                messageCount++;

            return;
        }

        if (MPI_Wtime() - start > DEADLOCK_TIMEOUT) {
            fprintf(stderr, "[DEADLOCK] Rank %d waiting for %d in %s\n", RANK, src, ctx);
            MPI_Abort(MPI_COMM_WORLD, 99);
        }
    }
}

// ============================================================================
// BFS Topology Builder (Rank 0 only)
// ============================================================================
void bfs_generate_topology(const std::string& graphFile, const std::string& topoFile)
{
    std::ifstream f(graphFile);
    if (!f.is_open()) {
        std::cerr << "[BFS] Cannot open graph file\n";
        MPI_Abort(MPI_COMM_WORLD, 300);
    }

    json j;
    f >> j;
    int n = j["nodes"];

    std::vector<std::vector<int>> g(n);
    for (auto& e : j["edges"]) {
        g[e["u"]].push_back(e["v"]);
        g[e["v"]].push_back(e["u"]);
    }

    std::vector<int> parent(n, -2);
    std::vector<std::vector<int>> children(n);

    parent[0] = -1;
    std::queue<int> q;
    q.push(0);

    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : g[u]) {
            if (parent[v] == -2) {
                parent[v] = u;
                children[u].push_back(v);
                q.push(v);
            }
        }
    }

    for (int i = 0; i < n; i++)
        if (parent[i] == -2) {
            std::cerr << "[BFS] Graph disconnected!\n";
            MPI_Abort(MPI_COMM_WORLD, 301);
        }

    json out;
    out["parent"] = parent;
    out["children"] = children;

    std::ofstream fout(topoFile);
    fout << out.dump(4);
}

// ============================================================================
// Load Graph (NO MST counting)
// ============================================================================
void loadGraph(const std::string& path)
{
    int n = 0, ec = 0;
    std::vector<int> buf;

    if (RANK == 0) {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "Cannot open graph\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        json j; f >> j;

        n = j["nodes"];
        ec = j["edges"].size();

        buf.reserve(ec * 3);
        for (auto& e : j["edges"]) {
            buf.push_back(e["u"]);
            buf.push_back(e["v"]);
            buf.push_back(e["w"]);
        }

        if (n != SIZE) {
            std::cerr << "[ERR] nodes != MPI ranks\n";
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ec, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (RANK != 0) buf.resize(ec * 3);
    if (ec * 3 > 0)
        MPI_Bcast(buf.data(), ec * 3, MPI_INT, 0, MPI_COMM_WORLD);

    edges.clear();
    edges.reserve(ec);

    for (int i = 0; i < ec; i++)
        edges.push_back({ buf[3 * i], buf[3 * i + 1], buf[3 * i + 2] });

    adj.assign(SIZE, {});
    for (auto& e : edges) {
        adj[e.u].push_back(e.v);
        adj[e.v].push_back(e.u);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// ============================================================================
// Load BFS Topology
// ============================================================================
void loadTopology(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Cannot open topology\n";
        MPI_Abort(MPI_COMM_WORLD, 400);
    }

    json j;
    f >> j;

    parentT = j["parent"].get<std::vector<int>>();
    childrenT = j["children"].get<std::vector<std::vector<int>>>();
}

void validateTopology()
{
    MPI_Barrier(MPI_COMM_WORLD);
    int rootCount = 0;

    for (int i = 0; i < SIZE; i++)
        if (parentT[i] == -1) rootCount++;

    if (rootCount != 1) {
        if (isRoot) std::cerr << "ERROR: BFS must have exactly one root\n";
        MPI_Abort(MPI_COMM_WORLD, 500);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// ============================================================================
// Helper
// ============================================================================
bool isLeaf() { return childrenT[RANK].empty(); }

// Fixed: avoid passing pointer into recv buffer (no aliasing)
void synchronizeFragments()
{
    int sendRoot = fragRoot[RANK];
    MPI_Allgather(&sendRoot, 1, MPI_INT,
        fragRoot.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int sendRank = fragRank[RANK];
    MPI_Allgather(&sendRank, 1, MPI_INT,
        fragRank.data(), 1, MPI_INT, MPI_COMM_WORLD);
}

// ============================================================================
// Edge name (for hashing)
// ============================================================================
uint64_t edgeName(int u, int v) {
    if (u > v) std::swap(u, v);
    return ((uint64_t)u << 32) ^ (uint64_t)v;
}

// ============================================================================
// DISC-Lite Hash Function
// ============================================================================
struct HashFunc {
    uint64_t a, b;
    uint64_t mod = (1ULL << 61) - 1;

    HashFunc() {
        // note: rand() has limited range, but this suffices for diversity per-rank
        a = 1 + (rand() % 2147483646);
        b = rand();
    }

    inline uint64_t hash(uint64_t x) const {
        __uint128_t r = (__uint128_t)a * x + b;
        r = (r >> 61) + (r & mod);
        if (r >= mod) r -= mod;
        return (uint64_t)r;
    }
};

// ============================================================================
// DISC-Lite Sampling FindMin
// ============================================================================
Edge findMinEdge(int fragRep)
{
    HashFunc H;
    std::vector<std::pair<uint64_t, Edge>> cand;

    for (auto& e : edges) {
        bool uIn = (fragRoot[e.u] == fragRep);
        bool vIn = (fragRoot[e.v] == fragRep);
        if (uIn == vIn) continue;

        uint64_t h = H.hash(edgeName(e.u, e.v));
        cand.push_back({ h, e });
    }

    if (cand.empty()) return { -1, -1, INT_MAX };

    std::vector<uint64_t> vals;
    vals.reserve(cand.size());
    for (auto& p : cand) vals.push_back(p.first);

    std::sort(vals.begin(), vals.end());

    uint64_t threshold =
        vals[std::max(0, int(vals.size() * 0.15))];

    Edge best = { -1, -1, INT_MAX };
    for (auto& p : cand) {
        if (p.first <= threshold) {
            if (p.second.w < best.w)
                best = p.second;
        }
    }

    return best;
}

// ============================================================================
// ONE MST PHASE
// ============================================================================
bool runPhase()
{
    if (isRoot) {
        std::cout << "\n===== Phase " << phaseCounter << " =====\n";
        for (int i = 0; i < SIZE; i++)
            std::cout << "Node " << i << ": frag=" << fragRoot[i]
            << " rank=" << fragRank[i] << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // -------------------------------------------------------------------------
    // STEP 1: Broadcast Rank-Request  (NOT COUNTED)
    // -------------------------------------------------------------------------
    {
        Msg m{}; m.type = MSG_RANK_REQUEST;

        if (isRoot) {
            for (int c : childrenT[RANK])
                safe_send(c, m, "RR bcast");
        }
        else {
            safe_recv(parentT[RANK], m, MSG_RANK_REQUEST, "RR recv");
            for (int c : childrenT[RANK])
                safe_send(c, m, "RR fwd");
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // -------------------------------------------------------------------------
    // STEP 2: Convergecast rank  (NOT COUNTED)
    // -------------------------------------------------------------------------
    int minRankVal = fragRank[RANK];
    for (int c : childrenT[RANK]) {
        Msg ch{};
        safe_recv(c, ch, MSG_RANK_REQUEST, "RR_UP");
        minRankVal = std::min(minRankVal, ch.a);
    }

    if (!isRoot) {
        Msg up{};
        up.type = MSG_RANK_REQUEST;
        up.a = minRankVal;
        safe_send(parentT[RANK], up, "RR_UP send");
    }
    else {
        fragRank[RANK] = minRankVal;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // -------------------------------------------------------------------------
    // STEP 3: Broadcast proceed (NOT COUNTED)
    // -------------------------------------------------------------------------
    {
        Msg pm{}; pm.type = MSG_PROCEED;
        pm.a = fragRank[0];

        if (isRoot) {
            for (int c : childrenT[RANK])
                safe_send(c, pm, "Proceed");
        }
        else {
            safe_recv(parentT[RANK], pm, MSG_PROCEED, "Proceed recv");
            for (int c : childrenT[RANK])
                safe_send(c, pm, "Proceed fwd");
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // -------------------------------------------------------------------------
    // STEP 4: Fragment leaders compute DISC-Lite min edge (LOCAL)
    // -------------------------------------------------------------------------
    Edge myBest = { -1, -1, INT_MAX };

    if (fragRoot[RANK] == RANK)
        myBest = findMinEdge(RANK);

    Msg prop{};
    prop.type = MSG_CONNECT;

    if (myBest.u != -1) {
        prop.a = myBest.u;
        prop.b = myBest.v;
        prop.c = myBest.w;
    }
    else {
        prop.a = prop.b = -1;
        prop.c = INT_MAX;
    }

    // -------------------------------------------------------------------------
    // STEP 5: Convergecast CONNECT proposals (COUNTED)
    // -------------------------------------------------------------------------
    Edge bestLocal = myBest;

    for (int c : childrenT[RANK]) {
        Msg child{};
        safe_recv(c, child, MSG_CONNECT, "CONNECT recv");

        if (child.a != -1) {
            Edge e = { child.a, child.b, child.c };
            if (e.w < bestLocal.w)
                bestLocal = e;
        }
    }

    if (!isRoot) {
        Msg up{};
        up.type = MSG_CONNECT;
        if (bestLocal.u != -1) {
            up.a = bestLocal.u;
            up.b = bestLocal.v;
            up.c = bestLocal.w;
        }
        else {
            up.a = up.b = -1;
            up.c = INT_MAX;
        }
        safe_send(parentT[RANK], up, "CONNECT up");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // -------------------------------------------------------------------------
    // STEP 6: Root does multi-merge
    // -------------------------------------------------------------------------
    std::vector<Edge> mergeSet;

    if (isRoot) {
        std::vector<Edge> proposals;
        if (bestLocal.u != -1)
            proposals.push_back(bestLocal);

        std::sort(proposals.begin(), proposals.end(),
            [](auto& a, auto& b) {return a.w < b.w;});

        std::vector<bool> used(SIZE, false);

        for (auto& e : proposals) {
            int fu = fragRoot[e.u];
            int fv = fragRoot[e.v];
            if (fu == fv) continue;

            if (!used[fu] && !used[fv]) {
                mergeSet.push_back(e);
                used[fu] = used[fv] = true;
            }
        }

        std::cout << "[Root] merges = " << mergeSet.size() << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // -------------------------------------------------------------------------
    // STEP 7: Broadcast merge count (COUNTED)
    // -------------------------------------------------------------------------
    int M = mergeSet.size();

    {
        Msg msg{};
        msg.type = MSG_CONNECT;
        msg.a = M;

        if (isRoot) {
            for (int c : childrenT[RANK])
                safe_send(c, msg, "MERGE size bcast");
        }
        else {
            safe_recv(parentT[RANK], msg, MSG_CONNECT, "MERGE size recv");
            M = msg.a;
            for (int c : childrenT[RANK])
                safe_send(c, msg, "MERGE size fwd");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // -------------------------------------------------------------------------
    // STEP 8: Send each merge triple (COUNTED)
    // -------------------------------------------------------------------------
    std::vector<std::tuple<int, int, int>> mergeItems(M);

    if (isRoot) {
        for (int i = 0; i < M; i++) {
            Edge e = mergeSet[i];
            int nf = std::min(fragRoot[e.u], fragRoot[e.v]);
            mergeItems[i] = { e.u, e.v, nf };
            mst_edges.push_back(e);
        }
    }

    for (int i = 0; i < M; i++) {
        Msg msg{};
        msg.type = MSG_CONNECT;

        if (isRoot) {
            msg.a = std::get<0>(mergeItems[i]);
            msg.b = std::get<1>(mergeItems[i]);
            msg.c = std::get<2>(mergeItems[i]);

            for (int c : childrenT[RANK])
                safe_send(c, msg, "MERGE triple");
        }
        else {
            safe_recv(parentT[RANK], msg, MSG_CONNECT, "MERGE triple recv");
            mergeItems[i] = { msg.a, msg.b, msg.c };

            for (int c : childrenT[RANK])
                safe_send(c, msg, "MERGE triple fwd");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // -------------------------------------------------------------------------
    // STEP 9: Apply merges locally
    // -------------------------------------------------------------------------
    for (auto& [u, v, nf] : mergeItems) {
        int fu = fragRoot[u];
        int fv = fragRoot[v];

        if (fragRoot[RANK] == fu || fragRoot[RANK] == fv)
            fragRoot[RANK] = nf;
    }

    synchronizeFragments();
    MPI_Barrier(MPI_COMM_WORLD);

    // -------------------------------------------------------------------------
    // STEP 10: DONE convergecast (NOT COUNTED)
    // -------------------------------------------------------------------------
    int newRank = fragRank[RANK];
    if (fragRoot[RANK] == RANK) newRank++;

    for (int c : childrenT[RANK]) {
        Msg d{};
        safe_recv(c, d, MSG_DONE, "DONE recv");
        newRank = std::min(newRank, d.a);
    }

    fragRank[RANK] = newRank;

    if (!isRoot) {
        Msg d{};
        d.type = MSG_DONE;
        d.a = newRank;
        safe_send(parentT[RANK], d, "DONE up");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (isRoot) return (M > 0);
    return true;
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &SIZE);

    isRoot = (RANK == 0);

    // seed rand() so HashFunc varies across ranks and runs
    srand((unsigned int)(time(nullptr) + RANK * 1315423911u));

    if (argc < 3) {
        if (isRoot)
            std::cout << "Usage: mpirun -np N ./mst graph.json topology.json\n";
        MPI_Finalize();
        return 0;
    }

    std::string graphFile = argv[1];
    std::string topoFile = argv[2];

    // BFS building
    if (isRoot) {
        std::cout << "\n[Rank0] Generating topology...\n";
        bfs_generate_topology(graphFile, topoFile);
        std::cout << "[Rank0] BFS complete.\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    loadGraph(graphFile);
    loadTopology(topoFile);
    validateTopology();

    fragRoot.resize(SIZE);
    fragRank.resize(SIZE);

    for (int i = 0; i < SIZE; i++) {
        fragRoot[i] = i;
        fragRank[i] = 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (isRoot)
        std::cout << "[Rank0] Starting MST...\n";

    auto start = Clock::now();

    COUNTING_ENABLED = true;

    phaseCounter = 0;
    int maxPhases = 2 * SIZE;

    while (phaseCounter < maxPhases) {
        phaseCounter++;

        bool merged = runPhase();

        synchronizeFragments();

        int myFrag = fragRoot[RANK];
        int localSame = 1;

        for (int i = 0; i < SIZE; i++)
            if (fragRoot[i] != myFrag)
                localSame = 0;

        int allSame;
        MPI_Allreduce(&localSame, &allSame, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        if (allSame == 1) {
            if (isRoot) std::cout << "MST complete.\n";
            break;
        }
    }

    auto end = Clock::now();
    double totalTime = std::chrono::duration<double>(end - start).count();

    COUNTING_ENABLED = false;

    long long totalMsgs = 0;
    MPI_Reduce(&messageCount, &totalMsgs, 1, MPI_LONG_LONG,
        MPI_SUM, 0, MPI_COMM_WORLD);

    if (isRoot) {
        double density = (double)edges.size() /
            (double)(SIZE * (SIZE - 1) / 2.0);

        std::ofstream out("topology_results.csv", std::ios::app);
        out << SIZE << ","
            << density << ","
            << totalTime << ","
            << totalMsgs << ","
            << phaseCounter << "\n";
        out.close();

        std::cout << "\n===== MST Construction Completed =====\n";
        std::cout << "Total time:      " << totalTime << " sec\n";
        std::cout << "Total messages:  " << totalMsgs << "\n";
        std::cout << "Phases:          " << phaseCounter << "\n";
        std::cout << "Density:         " << density << "\n";

        std::cout << "\nFinal Fragment State:\n";
        for (int i = 0; i < SIZE; i++)
            std::cout << "Node " << i << " → fragment "
            << fragRoot[i] << ", rank "
            << fragRank[i] << "\n";

        std::cout << "\nMST edges:\n";
        int valid = 0;
        for (auto& e : mst_edges) {
            if (e.u != -1) {
                std::cout << e.u << " -- " << e.v
                    << " (w=" << e.w << ")\n";
                valid++;
            }
        }

        std::cout << "Total MST edges: " << valid
            << " (expected " << SIZE - 1 << ")\n";
    }

    MPI_Finalize();
    return 0;
}
