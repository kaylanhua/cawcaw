import numpy as np
import matplotlib.pyplot as plt
import random
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
            num_connections = max(1, min(num_connections, 4))
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
            "id": i,
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

    plt.savefig('graph.png')

def generate_nodes(num_points, min_distance):
    points = generate_points(num_points, min_distance)
    tree = cKDTree(points)
    
    edges, angles = create_edges_and_angles(points, num_points, tree)
    
    k = 5
    distances, indices = tree.query(points, k=k+1)
    nodes = create_nodes(points, num_points, edges, angles, indices)

    return nodes, points, edges

import numpy as np
import matplotlib.pyplot as plt

def score_fn(anchor1, angles1, anchor2, angles2):
    def create_vector(anchor_point, angle_degrees):
        angle_radians = np.deg2rad(angle_degrees)
        vector_length = 1  

        # Calculate the vector components
        vector_x = vector_length * np.cos(angle_radians)
        vector_y = vector_length * np.sin(angle_radians)

        # Create the vector
        vector = np.array([vector_x, vector_y])
        return vector

    # Calculate the distance between the two anchors
    distance_between_anchors = np.linalg.norm(np.array(anchor2) - np.array(anchor1))

    lowest_score = float('inf')
    best_angle = None
    best_vector1 = None
    best_vector2 = None

    for angle1 in angles1:
        for angle2 in angles2:
            vector1 = create_vector(anchor1, angle1)
            vector2 = create_vector(anchor2, angle2)

            # Calculate the dot product and magnitudes of the vectors
            dot_product = np.dot(vector1, vector2)
            magnitude1 = np.linalg.norm(vector1)
            magnitude2 = np.linalg.norm(vector2)

            # Calculate the angle between the vectors
            angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))
            angle_degrees = np.rad2deg(angle_radians)

            # Normalize the angle to be between 0 and 180 degrees
            normalized_angle = min(angle_degrees, 180 - angle_degrees)
            
            # Calculate the angle score (higher score for angles closer to 0 or 180)
            angle_score = 1 - (normalized_angle / 90)
            
            # Calculate the distance score (higher score for shorter distances)
            max_distance = 10  # Assume a maximum distance for normalization
            distance_score = max(0, 1 - (distance_between_anchors / max_distance))
            
            # Combine the angle score and distance score
            probability_score = angle_score * distance_score

            if probability_score < lowest_score:
                lowest_score = probability_score
                best_angle = angle_degrees
                best_vector1 = vector1
                best_vector2 = vector2

    return best_angle, best_vector1, best_vector2, distance_between_anchors, lowest_score

def calculate_best_score(curr_nodes):
    # Calculate pairwise scores for starting nodes
    scores = []
    for i in range(len(curr_nodes)):
        for j in range(i + 1, len(curr_nodes)):
            node1 = curr_nodes[i]
            node2 = curr_nodes[j]
            
            angle, vec1, vec2, distance, score = score_fn(
                node1['position'], node1['angles'],
                node2['position'], node2['angles']
            )
            
            scores.append((score, node1, node2, angle))

    # Find the pair with the smallest score
    best_pair = min(scores, key=lambda x: x[0])
    best_score, best_node1, best_node2, best_angle = best_pair

    print(f"Best pair of nodes:")
    print(f"Node 1: {best_node1}")
    print(f"Node 2: {best_node2}")
    print(f"Score: {best_score}")
    print(f"Angle: {best_angle}")

    

    return best_score, best_node1, best_node2, best_angle

def main():
    num_points = 12
    min_distance = 0.5

    remaining_nodes, points, edges = generate_nodes(num_points, min_distance)

    print(remaining_nodes)

    # print_sample_nodes(remaining_nodes)
    plot_points_and_edges(points, edges)

    # sample 3 nodes
    curr_nodes = []
    for _ in range(3):
        if remaining_nodes:
            node = remaining_nodes.pop(random.randint(0, len(remaining_nodes) - 1))
            curr_nodes.append(node)

    while remaining_nodes:
        best_score, best_node1, best_node2 = calculate_best_score(curr_nodes)

        index_of_best_node = [node['id'] for node in remaining_nodes].index(best_node2['id'])
        new_node = remaining_nodes.pop(index_of_best_node)
        curr_nodes.append(new_node)

        print(f"curr_nodes: {curr_nodes}")


if __name__ == "__main__":
    main()
