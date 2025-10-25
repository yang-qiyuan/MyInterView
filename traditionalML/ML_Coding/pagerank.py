import numpy as np
import matplotlib.pyplot as plt

def pagerank(adj_matrix, d=0, max_iterations=1000, tolerance=1e-6):
    """
    Calculates PageRank scores (no damping) and records convergence speed.

    Args:
        adj_matrix (np.ndarray): Square adjacency matrix where A[i, j] = 1 if there's a link j -> i.
        d (float): Unused (kept for API compatibility). No damping here.
        max_iterations (int): Maximum number of iterations.
        tolerance (float): L1 convergence tolerance.

    Returns:
        scores (np.ndarray): Stationary distribution (PageRank without damping).
        deltas (np.ndarray): L1 step changes ||r_{t+1}-r_t||_1 over iterations.
    """
    # column out-degrees
    out_degree = np.sum(adj_matrix, axis=0)
    n = adj_matrix.shape[0]

    # handle dangling columns (out_degree == 0) by making them uniform
    dangling = (out_degree == 0)
    if np.any(dangling):
        adj_matrix = adj_matrix.copy()
        adj_matrix[:, dangling] = 1.0
        out_degree = np.sum(adj_matrix, axis=0)

    # column-stochastic transition matrix
    P = adj_matrix / out_degree

    # power iteration on column vector
    r = np.ones(n) / n
    deltas = []
    for _ in range(max_iterations):
        r_next = P @ r
        delta = np.linalg.norm(r_next - r, 1)
        deltas.append(delta)
        r = r_next
        if delta < tolerance:
            break

    return r, np.array(deltas)

# --- Example usage ---
if __name__ == "__main__":
    # Your example graph (kept exactly, using A.T to match j->i convention)
    adj_matrix = np.array([
        [0, 0, 1, 1, 0, 0],  # Links to node 0
        [0, 0, 0, 1, 1, 0],  # Links to node 1
        [0, 0, 0, 1, 0, 1],  # Links to node 2
        [0, 0, 0, 0, 0, 1],  # Links to node 3
        [0, 0, 0, 1, 0, 1],  # Links to node 4
        [1, 1, 0, 0, 0, 0]   # Links to node 5
    ]).T

    scores, deltas = pagerank(adj_matrix, d=0, max_iterations=5000, tolerance=1e-12)

    print("\nPageRank Scores:")
    for i, s in enumerate(scores):
        print(f"Node {i}: {s:.6f}")
    print(f"\nSum of scores: {np.sum(scores):.6f}")

    # plot speed of convergence (L1 step change)
    plt.figure(figsize=(6, 4))
    plt.semilogy(np.arange(1, len(deltas) + 1), deltas, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('L1 change')
    plt.title('PageRank: convergence speed')
    plt.tight_layout()
    plt.show()
