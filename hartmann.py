import numpy as np

A = np.array(
    [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]]
)

P = (
    np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    / 10000.0
)

alpha = np.array([1.0, 1.2, 3.0, 3.2])


def hartmann6(x):
    assert x.shape == (6,)
    inner_sum = np.sum(A * np.square(x - P), axis=-1)
    return -(np.sum(alpha * np.exp(-inner_sum), axis=-1))


def hartmann6_50(x, embedding_idx=[1, 7, 11, 23, 47, 33]):
    assert x.shape == (50,)
    return hartmann6(x[embedding_idx])
