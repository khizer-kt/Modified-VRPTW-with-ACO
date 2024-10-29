data["distance_matrix"] = [
    [0, 6, 9, 8, 7, 3, 6, 2, 3, 2, 6, 6, 4, 4, 5, 9, 7, 12, 8, 15, 10, 11, 7, 5, 9, 14, 6, 8, 7, 13],
    [6, 0, 8, 3, 2, 6, 8, 4, 8, 8, 13, 7, 5, 8, 12, 10, 14, 11, 9, 12, 14, 8, 10, 11, 8, 9, 12, 15, 10, 13],
    [9, 8, 0, 11, 10, 6, 3, 9, 5, 8, 4, 15, 14, 13, 9, 18, 9, 13, 7, 15, 8, 11, 12, 9, 14, 10, 11, 7, 8, 12],
    [8, 3, 11, 0, 1, 7, 10, 6, 10, 10, 14, 6, 7, 9, 14, 6, 16, 11, 15, 10, 12, 9, 8, 10, 6, 7, 8, 14, 12, 11],
    [7, 2, 10, 1, 0, 6, 9, 4, 8, 9, 13, 4, 6, 8, 12, 8, 14, 10, 12, 9, 8, 11, 7, 10, 12, 13, 14, 11, 10, 9],
    [3, 6, 6, 7, 6, 0, 2, 3, 2, 2, 7, 9, 7, 7, 6, 12, 8, 5, 4, 3, 7, 8, 9, 6, 10, 8, 12, 14, 11, 10],
    [6, 8, 3, 10, 9, 2, 0, 6, 2, 5, 4, 12, 10, 10, 6, 15, 5, 8, 9, 12, 7, 6, 11, 10, 12, 11, 8, 9, 6, 10],
    [2, 4, 9, 6, 4, 3, 6, 0, 4, 4, 8, 5, 4, 3, 7, 8, 10, 9, 7, 8, 10, 12, 9, 10, 7, 6, 5, 11, 12, 10],
    [3, 8, 5, 10, 8, 2, 2, 4, 0, 3, 4, 9, 8, 7, 3, 13, 6, 12, 11, 14, 9, 8, 10, 6, 5, 7, 12, 8, 10, 12],
    [2, 8, 8, 10, 9, 2, 5, 4, 3, 0, 4, 6, 5, 4, 3, 9, 5, 8, 12, 10, 14, 11, 13, 6, 5, 9, 10, 8, 14, 11],
    [6, 13, 4, 14, 13, 7, 4, 8, 4, 4, 0, 10, 9, 8, 4, 13, 4, 10, 7, 9, 12, 8, 10, 9, 11, 12, 13, 15, 14, 10],
    [6, 7, 15, 6, 4, 9, 12, 5, 9, 6, 10, 0, 1, 3, 7, 3, 10, 12, 14, 8, 9, 6, 5, 11, 12, 10, 8, 7, 8, 9],
    [4, 5, 14, 7, 6, 7, 10, 4, 8, 5, 9, 1, 0, 2, 6, 4, 8, 10, 11, 9, 8, 7, 9, 6, 7, 8, 9, 11, 10, 8],
    [4, 8, 13, 9, 8, 7, 10, 3, 7, 4, 8, 3, 2, 0, 4, 5, 6, 9, 11, 12, 10, 14, 9, 8, 10, 12, 15, 13, 10, 9],
    [5, 12, 9, 14, 12, 6, 6, 7, 3, 3, 4, 7, 6, 4, 0, 9, 2, 6, 5, 7, 8, 9, 10, 6, 8, 11, 9, 12, 10, 7],
    [9, 10, 18, 6, 8, 12, 15, 8, 13, 9, 13, 3, 4, 5, 9, 0, 9, 11, 10, 14, 15, 12, 11, 10, 9, 6, 5, 8, 10, 11],
    [7, 14, 9, 16, 14, 8, 5, 10, 6, 5, 4, 10, 8, 6, 2, 9, 0, 7, 8, 10, 12, 9, 10, 11, 8, 7, 6, 10, 14, 12],
    [12, 11, 13, 11, 10, 5, 8, 9, 12, 12, 10, 12, 10, 8, 9, 11, 7, 0, 7, 6, 5, 6, 10, 8, 9, 8, 10, 11, 14, 10],
    [8, 9, 7, 15, 12, 4, 9, 7, 11, 10, 12, 14, 11, 10, 11, 9, 8, 7, 0, 5, 6, 8, 9, 7, 8, 10, 9, 6, 7, 10],
    [15, 12, 15, 10, 9, 3, 12, 8, 14, 8, 9, 8, 8, 12, 9, 14, 10, 6, 5, 0, 4, 5, 7, 8, 12, 9, 11, 10, 14, 13],
    [10, 14, 8, 12, 8, 7, 7, 10, 9, 14, 12, 9, 10, 10, 8, 15, 12, 5, 6, 4, 0, 10, 9, 8, 7, 8, 10, 9, 10, 12],
    [11, 8, 11, 9, 11, 7, 6, 12, 8, 11, 14, 6, 9, 8, 14, 12, 9, 6, 8, 5, 10, 0, 4, 8, 6, 12, 10, 11, 10, 9],
    [7, 10, 12, 8, 7, 9, 11, 9, 10, 13, 9, 5, 9, 9, 9, 11, 10, 10, 9, 7, 9, 4, 0, 6, 8, 7, 5, 9, 12, 10],
    [5, 11, 9, 10, 10, 6, 10, 10, 6, 6, 11, 11, 6, 8, 6, 10, 11, 8, 7, 8, 8, 8, 6, 0, 12, 9, 10, 11, 9, 8],
    [9, 8, 14, 6, 12, 10, 12, 7, 5, 5, 11, 12, 7, 10, 8, 9, 8, 9, 8, 12, 7, 6, 8, 12, 0, 7, 10, 8, 7, 11],
    [14, 9, 10, 7, 13, 8, 11, 6, 7, 9, 12, 10, 8, 12, 11, 6, 7, 8, 10, 9, 8, 12, 7, 9, 7, 0, 5, 6, 7, 10],
    [6, 12, 11, 8, 14, 12, 8, 5, 12, 10, 13, 8, 9, 15, 9, 5, 6, 10, 9, 11, 10, 10, 5, 10, 10, 5, 0, 11, 9, 8],
    [8, 15, 7, 14, 11, 14, 9, 11, 8, 8, 15, 7, 11, 13, 12, 8, 10, 11, 6, 10, 9, 11, 9, 11, 8, 6, 11, 0, 12, 11],
    [7, 10, 8, 12, 10, 11, 6, 12, 10, 14, 14, 8, 10, 10, 10, 11, 14, 14, 7, 14, 10, 9, 12, 9, 7, 7, 9, 12, 0, 10],
    [13, 13, 12, 11, 9, 10, 10, 10, 12, 11, 10, 9, 8, 9, 7, 11, 10, 10, 10, 13, 12, 9, 10, 8, 11, 10, 8, 11, 10, 0]
]

data["time_windows"] = [
    [0, 10], [2, 12], [3, 13], [0, 10], [2, 15], [0, 9], [1, 11], [0, 12], [1, 12], [0, 8],
    [4, 15], [2, 8], [1, 10], [0, 10], [2, 15], [0, 10], [3, 12], [2, 12], [3, 13], [2, 10],
    [0, 15], [1, 10], [3, 11], [2, 12], [0, 10], [0, 12], [3, 15], [4, 12], [1, 9], [2, 10]
]
