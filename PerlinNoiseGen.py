import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from opensimplex import OpenSimplex
import math

# Heightmap size
width = 512
height = 512


"""
STEP 1: CREATE A HEIGHTMAP USING PERLIN NOISE
"""


def create_heightmap(width: int, height: int, scale: float = 256.0, seed: int = 42):
    # Create a 2D array of floats
    heightmap = np.zeros((width, height), dtype=float)

    # Create a Perlin noise object
    noise_generator = OpenSimplex(seed=seed)

    # Generate Perlin noise and assign it to the heightmap
    for i in range(height):
        for j in range(width):
            # Generate multiple octaves of noise
            for octave in range(6):  # Adjust the number of octaves as needed
                frequency = 2 ** octave
                amplitude = 1 / frequency
                heightmap[i][j] += (1 - abs(noise_generator.noise2(i / scale * frequency, j / scale * frequency))) * amplitude



    # Normalize the heightmap to the range [0, 1]
    heightmap -= heightmap.min()
    heightmap /= heightmap.max()

    # Invert the heightmap values to create ridges at high values
    heightmap = 1 - heightmap

    return heightmap


heightmap = create_heightmap(width, height, seed=56456)

plt.figure(figsize=(width / 100, height / 100)) 
plt.imshow(heightmap, cmap='gray', interpolation='none')
plt.axis('off')  # Turn off axis
plt.show()