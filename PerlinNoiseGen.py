import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from opensimplex import OpenSimplex
import math

# Heightmap size
width = 128
height = 128


"""
STEP 1: CREATE A HEIGHTMAP USING PERLIN NOISE
"""


def create_heightmap(width: int, height: int, scale: float = 32.0, seed: int = 42):
    # Create a 2D array of floats
    heightmap = np.zeros((width, height), dtype=float)

    # Create a Perlin noise object
    noise_generator = OpenSimplex(seed=seed)

    # Generate Perlin noise and assign it to the heightmap
    for i in range(height):
        for j in range(width):
            heightmap[i][j] = noise_generator.noise2(i / scale, j / scale)
            heightmap[i][j] += noise_generator.noise2(i / scale / 2, j / scale / 2) * 0.5
            heightmap[i][j] += noise_generator.noise2(i / (scale / 4), j / (scale / 4)) * 0.25
            heightmap[i][j] += noise_generator.noise2(i / (scale / 8), j / (scale / 8)) * 0.125
            heightmap[i][j] += noise_generator.noise2(i / (scale / 16), j / (scale / 16)) * 0.0625
            heightmap[i][j] += noise_generator.noise2(i / (scale / 32), j / (scale / 32)) * 0.03125

    # Normalize the heightmap to the range [0, 1]
    heightmap -= heightmap.min()
    heightmap /= heightmap.max()



    return heightmap


heightmap = create_heightmap(width, height, seed=56456)

plt.figure(figsize=(width / 100, height / 100)) 
plt.imshow(heightmap, cmap='gray', interpolation='none')
plt.axis('off')  # Turn off axis
plt.show()