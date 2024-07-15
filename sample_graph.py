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
        neighbor_indices = [end if start == i else start for start, end in edges if start == i or end == i]
        node = {
            "id": i,
            'position': points[i],
            'neighbors': neighbor_indices,
            'angles': angles[i][:len(neighbor_indices)]
        }
        nodes.append(node)
    return nodes

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

class VectorCache():
    def __init__(self, nodes):
        vectors = set()
        for node in nodes:
            print(node)
            print(f"node['angles']: {node['angles']}")
            for angle in node['angles']:
                vectors.add((tuple(node['position']), angle))

        self.vectors = vectors

    def get_vectors_with_position(self, position):
        return [vector for vector in self.vectors if np.allclose(vector[0], position, atol=1e-8)]

    def add_vector(self, vector):
        position, angle = vector
        for v in self.vectors:
            if np.allclose(v[0], position, atol=1e-8) and np.isclose(v[1], angle, atol=1e-8):
                return  # Vector already exists, don't add a duplicate
        self.vectors.add(vector)

    def remove_vector(self, vector):
        position, angle = vector
        for v in self.vectors:
            if np.allclose(v[0], position, atol=1e-8) and np.isclose(v[1], angle, atol=1e-8):
                self.vectors.remove(v)
                return
        print(f"Warning: Vector {vector} not found in the set")

    def calculate_best_pair(self):
        best_score = float('inf')
        best_pair = None
        iterable_vectors = list(self.vectors)
        for i in range(len(iterable_vectors)):
            for j in range(i + 1, len(iterable_vectors)):
                best_angle, best_vector1, best_vector2, distance_between_anchors, score = score_fn(
                    iterable_vectors[i][0], [iterable_vectors[i][1]],
                    iterable_vectors[j][0], [iterable_vectors[j][1]]
                )
                if score < best_score:
                    best_score = score
                    best_pair = (iterable_vectors[i], iterable_vectors[j])
        return best_pair

    def __add__(self, other):
        if isinstance(other, VectorCache):
            self.vectors = self.vectors.union(other.vectors)
        elif isinstance(other, set):
            self.vectors = self.vectors.union(other)
        else:
            raise TypeError("Unsupported operand type for +: '{}' and '{}'".format(type(self).__name__, type(other).__name__))
        return self
    
    def __sub__(self, other):
        if isinstance(other, VectorCache):
            self.vectors = self.vectors.difference(other.vectors)
        elif isinstance(other, set):
            self.vectors = self.vectors.difference(other)
        return self  # Add this line to return the modified object

    def __len__(self):
        return len(self.vectors)
    
    def __str__(self) -> str:
        return_str = ""
        for vector in self.vectors:
            return_str += f"{vector}\n"
        return return_str
    

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

    loc_to_idx = {}
    for i, node in enumerate(nodes):
        loc_to_idx[tuple(node['position'])] = i

    return nodes, points, edges, loc_to_idx

import numpy as np
import matplotlib.pyplot as plt
import copy

def main():
    num_points = 12
    min_distance = 0.5

    all_nodes, points, edges, loc_to_idx = generate_nodes(num_points, min_distance)

    remaining_nodes = copy.deepcopy(all_nodes)

    print(remaining_nodes)
    remaining_vector_cache = VectorCache(remaining_nodes)
    final_length = len(remaining_vector_cache)
    cur_vector_cache = VectorCache([])

    print_sample_nodes(remaining_nodes)
    plot_points_and_edges(points, edges)

    # sample 3 nodes
    curr_nodes = []
    for _ in range(3):
        if remaining_nodes:
            node = remaining_nodes.pop(random.randint(0, len(remaining_nodes) - 1))
            print(node)
            curr_nodes.append(node)
            cur_vector_cache += VectorCache([node])
            remaining_vector_cache -= VectorCache([node])

    while len(cur_vector_cache) < final_length:
        vec1, vec2 = cur_vector_cache.calculate_best_pair()

        cur_node = all_nodes[loc_to_idx[vec1[0]]]
        cur_angle_idx = cur_node['angles'].index(vec1[1])
        next_node = cur_node['neighbors'][cur_angle_idx]
        # add all new nodes
        cur_vector_cache += VectorCache([next_node])
        # remove overlap old nodes
        cur_vector_cache.remove_vector(vec1)
        
        # remove from remaining
        remaining_vector_cache -= VectorCache([next_node])
        remaining_vector_cache.remove_vector(vec1)

        # print(f"remaining vector: {remaining_vector_cache}")
        # print(f"cur_vector_cache: {cur_vector_cache}")
        # print(cur_vector_cache)
        print(f"\nlen(remaining_vector_cache): {len(remaining_vector_cache)}")
        print(f"len(cur_vector_cache): {len(cur_vector_cache)}")


if __name__ == "__main__":
    main()
