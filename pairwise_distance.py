import numpy as np
import time


X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])


def loop_distance(X):
    """Calculate pairwise distances using loops."""
    n = len(X)
    ans = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            ans[i][j] = np.sqrt(np.sum((X[i] - X[j]) ** 2))

    return ans


def vector_distance(X):
    """Calculate pairwise distances using NumPy broadcasting."""
    diff = X[:, None] - X
    ans = np.sqrt(np.sum(diff ** 2, axis=2))
    return ans


def main():
    """Compare loop and vectorized pairwise distance methods."""
    start = time.perf_counter()
    a = loop_distance(X)
    end = time.perf_counter()
    loop_time = end - start

    start = time.perf_counter()
    b = vector_distance(X)
    end = time.perf_counter()
    vector_time = end - start

    print("Loop Method")
    print(a)

    print("\nVector Method")
    print(b)

    print("\nSame Output:", np.allclose(a, b))

    print("\nLoop Time:", loop_time)
    print("Vector Time:", vector_time)


if __name__ == "__main__":
    main()