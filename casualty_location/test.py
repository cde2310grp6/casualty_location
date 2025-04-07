import numpy as np
from collections import deque

def find_frontiers_test(map_array, robot_position, ignored_frontiers=set()):
    frontiers = []
    rows, cols = map_array.shape
    visited = np.zeros_like(map_array, dtype=bool)
    queue = deque()

    robot_row, robot_col = robot_position
    print(f"Robot starts at ({robot_row}, {robot_col}) with value {map_array[robot_row, robot_col]}")
    queue.append((robot_row, robot_col))
    visited[robot_row, robot_col] = True

    while queue:
        r, c = queue.popleft()

        if map_array[r, c] == 0:
            r_min = max(0, r - 1)
            r_max = min(rows, r + 2)
            c_min = max(0, c - 1)
            c_max = min(cols, c + 2)
            neighbors = map_array[r_min:r_max, c_min:c_max].flatten()

            if 100 in neighbors and (r, c) not in ignored_frontiers:
                frontiers.append((r, c))
                print(f"Frontier found at ({r}, {c})")

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not visited[nr, nc] and (map_array[nr, nc] == 0 or (0 < map_array[nr, nc] < 50)):
                        visited[nr, nc] = True
                        queue.append((nr, nc))

    print(f"Total frontiers found: {len(frontiers)}")
    return frontiers


# 🔧 Define a test map
test_map = np.array([
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    [100, 0,    0,   0,   100, -1,  -1,  -1,  -1, 100],
    [100, 0,    100, 0,   100, -1,  100, 100, -1, 100],
    [100, 0,    0,   0,   100, -1,  0,   0,   -1, 100],
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    [100, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 100],
    [100, -1,  0,    0,   0,   100, 100, -1,  -1, 100],
    [100, -1,  0,   100, 0,   -1,  -1,  -1,  -1, 100],
    [100, -1,  0,    0,  0,   -1,  -1,  -1,  -1, 100],
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
], dtype=int)

# 🟢 Simulate robot near frontiers
robot_position = (2, 1)  # On a free cell next to occupied

# 🚀 Run the test
find_frontiers_test(test_map, robot_position)
