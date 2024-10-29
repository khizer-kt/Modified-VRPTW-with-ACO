import random

def generate_distance_matrix(num_nodes, max_distance=20):
    """Generates a symmetric distance matrix with random distances."""
    distance_matrix = [[0 if i == j else random.randint(1, max_distance) for j in range(num_nodes)] for i in range(num_nodes)]
    
    # Making the matrix symmetric
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance_matrix[j][i] = distance_matrix[i][j]
    
    return distance_matrix

def generate_time_windows(num_nodes, max_time=20):
    """Generates random time windows for each node."""
    time_windows = []
    for _ in range(num_nodes):
        start = random.randint(0, max_time - 5)
        end = random.randint(start + 1, max_time)
        time_windows.append([start, end])
    
    return time_windows

# Example usage
num_nodes = 50  # Specify the desired number of nodes
distance_matrix = generate_distance_matrix(num_nodes)
time_windows = generate_time_windows(num_nodes)

# Display the generated matrices
print("Distance Matrix:")
for row in distance_matrix:
    print(row)

print("\nTime Windows:")
for tw in time_windows:
    print(tw)
