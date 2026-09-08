#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

const int INFINITY_WEIGHT = 100000;

double program_start_time;

int test_message_pending = 0;
int connect_message_pending = 0;
int report_message_pending = 0;

// Message type constants
const int MSG_CONNECT = 1;
const int MSG_INITIATE = 2;
const int MSG_TEST = 3;
const int MSG_ACCEPT_REJECT = 4;
const int MSG_REPORT = 5;
const int MSG_CHANGE_ROOT = 6;
const int MSG_TERMINATE = 7;

// Performance metrics
int total_messages_sent = 0;
int total_messages_received = 0;

queue<pair<int, vector<int>>> deferred_messages;
vector<tuple<int, int, int>> adjacency_list;

#define EDGE_WEIGHT(idx) get<0>(adjacency_list[idx])
#define EDGE_TYPE(idx) get<1>(adjacency_list[idx])
#define EDGE_NODE(idx) get<2>(adjacency_list[idx])

int fragment_level = 0;
int fragment_identifier = 0;
int parent_node = 0;
int current_state = 0;
int process_id;
int neighbor_count;
int minimum_edge_weight;
int minimum_edge_node;
int reports_received;
int test_target_node;
bool should_terminate = false;

void read_graph_input(string filename) {
    ifstream file_stream(filename);
    int total_nodes;
    file_stream >> total_nodes;
    
    for (int i = 0; i < process_id; ++i) {
        for (int j = 0; j < total_nodes; ++j) {
            int value;
            file_stream >> value;
        }
    }

    for (int i = 0; i < total_nodes; ++i) {
        int value;
        file_stream >> value;
        if (value != INFINITY_WEIGHT) {
            adjacency_list.push_back(make_tuple(value, 0, i));
        }
    }

    for (int i = process_id + 1; i < total_nodes; ++i) {
        for (int j = 0; j < total_nodes; ++j) {
            int value;
            file_stream >> value;
        }
    }
    
    neighbor_count = adjacency_list.size();
    sort(adjacency_list.begin(), adjacency_list.end());
}

void start_algorithm() {
    EDGE_TYPE(0) = 1;
    fragment_level = 0;
    current_state = 2;
    reports_received = 0;
    connect_message_pending = 0;
    report_message_pending = 0;

    if (!connect_message_pending) {
        connect_message_pending = 1;
        int level_data = 0;
        MPI_Request request;
        MPI_Isend(&level_data, 1, MPI_INT, EDGE_NODE(0), MSG_CONNECT, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        total_messages_sent++;
    }
}

void handle_connect_message(int sender_node, int sender_level) {
    int edge_index = 0;
    while (edge_index < neighbor_count) {
        if (EDGE_NODE(edge_index) == sender_node) {
            break;
        }
        ++edge_index;
    }

    if (sender_level < fragment_level) {
        EDGE_TYPE(edge_index) = 1;
        MPI_Request request;
        int init_data[3] = {fragment_level, fragment_identifier, current_state};
        MPI_Isend(&init_data, 3, MPI_INT, sender_node, MSG_INITIATE, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        total_messages_sent++;
    } else if (EDGE_TYPE(edge_index) == 0) {
        vector<int> msg_args(2);
        msg_args[0] = sender_node;
        msg_args[1] = sender_level;
        deferred_messages.push({MSG_CONNECT, msg_args});
    } else {
        MPI_Request request;
        int init_data[3] = {fragment_level + 1, EDGE_WEIGHT(edge_index), 1};
        MPI_Isend(&init_data, 3, MPI_INT, sender_node, MSG_INITIATE, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        total_messages_sent++;
    }
}

void send_report_to_parent() {
    int branch_children = 0;
    for (int i = 0; i < neighbor_count; ++i) {
        if (EDGE_TYPE(i) == 1 && EDGE_NODE(i) != parent_node) {
            ++branch_children;
        }
    }
    
    if (reports_received == branch_children && test_target_node == -1 && !report_message_pending) {
        report_message_pending = 1;
        current_state = 2;
        MPI_Request request;
        int weight_data = minimum_edge_weight;
        MPI_Isend(&weight_data, 1, MPI_INT, parent_node, MSG_REPORT, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        total_messages_sent++;
    }
}

void search_minimum_edge() {
    int edge_found = 0;
    for (int i = 0; i < neighbor_count; ++i) {
        if (EDGE_TYPE(i) == 0 && !test_message_pending) {
            test_message_pending = 1;
            edge_found = 1;
            test_target_node = EDGE_NODE(i);
            MPI_Request request;
            int test_data[2] = {fragment_level, fragment_identifier};
            MPI_Isend(&test_data, 2, MPI_INT, test_target_node, MSG_TEST, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            total_messages_sent++;
            break;
        }
    }
    
    if (!edge_found) {
        test_target_node = -1;
        send_report_to_parent();
    }
}

void handle_initiate_message(int sender, int new_level, int new_name, int new_state) {
    fragment_level = new_level;
    fragment_identifier = new_name;
    current_state = new_state;
    parent_node = sender;
    minimum_edge_node = -1;
    minimum_edge_weight = INFINITY_WEIGHT;
    test_target_node = -1;
    connect_message_pending = 0;
    report_message_pending = 0;

    for (int i = 0; i < neighbor_count; ++i) {
        if (EDGE_TYPE(i) == 1 && EDGE_NODE(i) != parent_node) {
            MPI_Request request;
            int init_data[3] = {fragment_level, fragment_identifier, current_state};
            MPI_Isend(&init_data, 3, MPI_INT, EDGE_NODE(i), MSG_INITIATE, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            total_messages_sent++;
        }
    }

    if (current_state == 1) {
        reports_received = 0;
        search_minimum_edge();
    }
}

void perform_change_root() {
    int edge_index = 0;
    while (edge_index < neighbor_count) {
        if (EDGE_NODE(edge_index) == minimum_edge_node) {
            break;
        }
        ++edge_index;
    }

    if (EDGE_TYPE(edge_index) == 1) {
        MPI_Request request;
        int dummy_data = 0;
        MPI_Isend(&dummy_data, 1, MPI_INT, minimum_edge_node, MSG_CHANGE_ROOT, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        total_messages_sent++;
    } else {
        EDGE_TYPE(edge_index) = 1;
        if (!connect_message_pending) {
            connect_message_pending = 1;
            int level_data = fragment_level;
            MPI_Request request;
            MPI_Isend(&level_data, 1, MPI_INT, minimum_edge_node, MSG_CONNECT, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            total_messages_sent++;
        }
    }
}

void handle_report_message(int sender, int reported_weight) {
    if (sender != parent_node) {
        if (reported_weight < minimum_edge_weight) {
            minimum_edge_weight = reported_weight;
            minimum_edge_node = sender;
        }
        reports_received += 1;
        send_report_to_parent();
    } else {
        if (current_state == 1) {
            vector<int> msg_args(2);
            msg_args[0] = sender;
            msg_args[1] = reported_weight;
            deferred_messages.push({MSG_REPORT, msg_args});
        } else if (reported_weight > minimum_edge_weight) {
            perform_change_root();
        } else if (reported_weight == minimum_edge_weight && minimum_edge_weight == INFINITY_WEIGHT) {
            int termination_signal = 1;
            should_terminate = true;
            for (int i = 0; i < neighbor_count; ++i) {
                if (EDGE_TYPE(i) == 1) {
                    MPI_Request request;
                    MPI_Isend(&termination_signal, 1, MPI_INT, EDGE_NODE(i), MSG_TERMINATE, MPI_COMM_WORLD, &request);
                    MPI_Wait(&request, MPI_STATUS_IGNORE);
                    total_messages_sent++;
                }
            }
        }
    }
}

void handle_test_message(int sender, int sender_level, int sender_name) {
    int edge_index = 0;
    while (edge_index < neighbor_count) {
        if (EDGE_NODE(edge_index) == sender) {
            break;
        }
        ++edge_index;
    }

    if (sender_level > fragment_level) {
        vector<int> msg_args(3);
        msg_args[0] = sender;
        msg_args[1] = sender_level;
        msg_args[2] = sender_name;
        deferred_messages.push({MSG_TEST, msg_args});
    } else if (sender_name == fragment_identifier) {
        if (EDGE_TYPE(edge_index) == 0) {
            EDGE_TYPE(edge_index) = -1;
        }
        if (EDGE_NODE(edge_index) != test_target_node) {
            int reject_signal = -1;
            MPI_Request request;
            MPI_Isend(&reject_signal, 1, MPI_INT, sender, MSG_ACCEPT_REJECT, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            total_messages_sent++;
        } else {
            search_minimum_edge();
        }
    } else {
        int accept_signal = 1;
        MPI_Request request;
        MPI_Isend(&accept_signal, 1, MPI_INT, sender, MSG_ACCEPT_REJECT, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        total_messages_sent++;
    }
}

void handle_accept_reject_message(int sender, int response) {
    int edge_index = 0;
    while (edge_index < neighbor_count) {
        if (EDGE_NODE(edge_index) == sender) {
            break;
        }
        ++edge_index;
    }
    
    if (response == -1) {
        if (EDGE_TYPE(edge_index) == 0) {
            EDGE_TYPE(edge_index) = -1;
        }
        search_minimum_edge();
    } else {
        test_target_node = -1;
        if (EDGE_WEIGHT(edge_index) < minimum_edge_weight) {
            minimum_edge_weight = EDGE_WEIGHT(edge_index);
            minimum_edge_node = sender;
        }
        send_report_to_parent();
    }
}

int main(int argc, char **argv) {
    int total_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    read_graph_input(argv[1]);
    MPI_Barrier(MPI_COMM_WORLD);
    program_start_time = MPI_Wtime();

    start_algorithm();
    
    int idle_iterations = 0;
    const int MAX_IDLE = 1000;
    
    while (1) {
        int message_available = 0;
        MPI_Status status_info;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &message_available, &status_info);
        
        if (message_available) {
            idle_iterations = 0;
            total_messages_received++;
            
            if (status_info.MPI_TAG == MSG_CONNECT) {
                int message_data;
                MPI_Status recv_status;
                MPI_Recv(&message_data, 1, MPI_INT, MPI_ANY_SOURCE, MSG_CONNECT, MPI_COMM_WORLD, &recv_status);
                handle_connect_message(recv_status.MPI_SOURCE, message_data);
            } else if (status_info.MPI_TAG == MSG_INITIATE) {
                int message_data[3];
                MPI_Status recv_status;
                MPI_Recv(&message_data, 3, MPI_INT, MPI_ANY_SOURCE, MSG_INITIATE, MPI_COMM_WORLD, &recv_status);
                handle_initiate_message(recv_status.MPI_SOURCE, message_data[0], message_data[1], message_data[2]);
            } else if (status_info.MPI_TAG == MSG_TEST) {
                int message_data[2];
                MPI_Status recv_status;
                MPI_Recv(&message_data, 2, MPI_INT, MPI_ANY_SOURCE, MSG_TEST, MPI_COMM_WORLD, &recv_status);
                handle_test_message(recv_status.MPI_SOURCE, message_data[0], message_data[1]);
            } else if (status_info.MPI_TAG == MSG_ACCEPT_REJECT) {
                test_message_pending = 0;
                int message_data;
                MPI_Status recv_status;
                MPI_Recv(&message_data, 1, MPI_INT, MPI_ANY_SOURCE, MSG_ACCEPT_REJECT, MPI_COMM_WORLD, &recv_status);
                handle_accept_reject_message(recv_status.MPI_SOURCE, message_data);
            } else if (status_info.MPI_TAG == MSG_REPORT) {
                int message_data;
                MPI_Status recv_status;
                MPI_Recv(&message_data, 1, MPI_INT, MPI_ANY_SOURCE, MSG_REPORT, MPI_COMM_WORLD, &recv_status);
                handle_report_message(recv_status.MPI_SOURCE, message_data);
            } else if (status_info.MPI_TAG == MSG_CHANGE_ROOT) {
                int message_data;
                MPI_Status recv_status;
                MPI_Recv(&message_data, 1, MPI_INT, MPI_ANY_SOURCE, MSG_CHANGE_ROOT, MPI_COMM_WORLD, &recv_status);
                perform_change_root();
            } else if (status_info.MPI_TAG == MSG_TERMINATE) {
                int message_data;
                MPI_Status recv_status;
                MPI_Recv(&message_data, 1, MPI_INT, MPI_ANY_SOURCE, MSG_TERMINATE, MPI_COMM_WORLD, &recv_status);
                for (int i = 0; i < neighbor_count; ++i) {
                    if (EDGE_TYPE(i) == 1 && EDGE_NODE(i) != recv_status.MPI_SOURCE) {
                        int termination_signal = 0;
                        MPI_Request request;
                        MPI_Isend(&termination_signal, 1, MPI_INT, EDGE_NODE(i), MSG_TERMINATE, MPI_COMM_WORLD, &request);
                        MPI_Wait(&request, MPI_STATUS_IGNORE);
                        total_messages_sent++;
                    }
                }
                break;
            }
        }
        
        // Process deferred messages - but limit processing to avoid infinite loops
        if (deferred_messages.size() > 0 && !message_available) {
            pair<int, vector<int>> deferred_msg = deferred_messages.front();
            deferred_messages.pop();
            
            if (deferred_msg.first == MSG_CONNECT) {
                handle_connect_message(deferred_msg.second[0], deferred_msg.second[1]);
            } else if (deferred_msg.first == MSG_REPORT) {
                handle_report_message(deferred_msg.second[0], deferred_msg.second[1]);
            } else if (deferred_msg.first == MSG_TEST) {
                handle_test_message(deferred_msg.second[0], deferred_msg.second[1], deferred_msg.second[2]);
            }
        } else if (!message_available && deferred_messages.empty()) {
            idle_iterations++;
            if (idle_iterations > MAX_IDLE) {
                // Possible deadlock - force termination
                break;
            }
            // Small sleep to prevent busy waiting
            usleep(100);
        }
        
        if (should_terminate) {
            break;
        }
    }
    
    double end_time = MPI_Wtime();
    double execution_time = end_time - program_start_time;
    
    // Count total edges in the graph
    int total_edges = 0;
    for (int i = 0; i < neighbor_count; ++i) {
        if (EDGE_NODE(i) > process_id) {
            total_edges++;
        }
    }
    
    // Output performance metrics in CSV format
    printf("METRICS,%d,%lf,%d,%d,%d\n", 
           process_id, execution_time, total_messages_sent, 
           total_messages_received, total_edges);
    
    // Output MST edges
    for (int i = 0; i < neighbor_count; ++i) {
        if (EDGE_TYPE(i) == 1 && EDGE_NODE(i) > process_id) {
            printf("MST,%d,%d,%d\n", process_id, EDGE_NODE(i), EDGE_WEIGHT(i));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}