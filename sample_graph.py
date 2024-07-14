import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def generate_points(num_points, min_distance):
    points = []
    while len(points) < num_points:
        new_point = np.random.uniform(0, 10, 2)
        if not points or np.min([np.linalg.norm(new_point - p) for p in points]) >= min_distance:
            points.append(new_point)
    return np.array(points)

def create_edges_and_angles(points, num_points, tree):
    edges = []
    angles = {}
    k = 5
    distances, indices = tree.query(points, k=k+1)
    for i in range(num_points):
        connections = 0
        angles[i] = []
        for j in indices[i, 1:]:
            num_connections = int(np.random.normal(2, 0.5))
            num_connections = max(0, min(num_connections, 4))
            if connections < num_connections:
                edges.append((i, j))
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                angle = np.arctan2(dy, dx)
                angle_deg = (np.degrees(angle) + 360) % 360
                angles[i].append(angle_deg)
                connections += 1
    return edges, angles

def create_nodes(points, num_points, edges, angles, indices):
    nodes = []
    for i in range(num_points):
        neighbor_indices = [j for j, (start, end) in enumerate(edges) if start == i or end == i]
        valid_neighbors = [indices[i, j+1] for j in neighbor_indices if j+1 < len(indices[i])]
        node = {
            'position': points[i],
            'neighbors': valid_neighbors,
            'angles': angles[i][:len(valid_neighbors)]
        }
        nodes.append(node)
    return nodes

def print_sample_nodes(nodes):
    print("Sample nodes:")
    for i in range(min(5, len(nodes))):
        print(f"Node {i}:")
        print(f"  Position: {nodes[i]['position']}")
        print(f"  Neighbors: {nodes[i]['neighbors']}")
        print(f"  Angles: {nodes[i]['angles']}")
        print()

def plot_points_and_edges(points, edges):
    plt.figure(figsize=(8, 8))
    x, y = points[:, 0], points[:, 1]
    plt.scatter(x, y, alpha=0.6)

    for edge in edges:
        plt.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], 'k-', alpha=0.2)

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.title('Connected Points in a 10x10 Square (Min Distance: 0.5)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def generate_nodes(num_points, min_distance):
    points = generate_points(num_points, min_distance)
    tree = cKDTree(points)
    
    edges, angles = create_edges_and_angles(points, num_points, tree)
    
    k = 5
    distances, indices = tree.query(points, k=k+1)
    nodes = create_nodes(points, num_points, edges, angles, indices)

    return nodes, points, edges

def main():
    num_points = 12
    min_distance = 0.5

    nodes, points, edges = generate_nodes(num_points, min_distance)

    print_sample_nodes(nodes)
    plot_points_and_edges(points, edges)

if __name__ == "__main__":
    main()
