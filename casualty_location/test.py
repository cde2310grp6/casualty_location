import numpy as np
import cv2
import matplotlib.pyplot as plt


CASUALTY_COUNT = 3

grid = np.array([
    [30, 30, 30, 20, 20],
    [30, 30, 30, 20, 90],
    [20, 20, 20, 20, 90],
    [20, 30, 20, 20, 90],
    [90, 90, 90, 20, 20]
], dtype=np.float32)


def threshold(grid, thresh):
    binary = np.zeros_like(grid, dtype=np.uint8)
    rows, cols = grid.shape
    for row in range(rows):
        for col in range(cols):
            binary[row][col] = 255 if grid[row][col] >= thresh else 0
    return binary

def find_centroids(grid):
    # Remove unassigned cells (just for your case, but be careful with overwriting real data)
    grid = np.where(grid > 90, 0, grid)

    upper_threshold = 100.0
    lower_threshold = 0.0
    middle_threshold = (upper_threshold + lower_threshold) / 2
    tries = 0
    contours = []

    while tries < 100:
        binary = threshold(grid, middle_threshold)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == CASUALTY_COUNT:
            break
        elif len(contours) < CASUALTY_COUNT:
            upper_threshold = middle_threshold
        else:  # more than needed
            lower_threshold = middle_threshold

        middle_threshold = (upper_threshold + lower_threshold) / 2
        tries += 1

    if tries >= 100:
        print("❌ Failed to find casualties.")
        return []

    print("✅ Found casualties.")

    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))

    print("Centroids:", centroids)
    return centroids

centroids = find_centroids(grid)
