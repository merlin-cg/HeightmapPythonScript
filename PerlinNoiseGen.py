import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from opensimplex import OpenSimplex
import math
import platform

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

plt.savefig("/home/s5609424/Pictures/heightmap.png", bbox_inches='tight', pad_inches=0)  # Save the image without extra padding
# plt.show()

noise_path = '/home/s5609424/Pictures/heightmap.png'


# Generate plane and assign heightmap to displacement shader
def genPlane():
    plane = cmds.polyPlane(width=10, height=10, subdivisionsX=10, subdivisionsY=10)[0]
    
    lambert_shader = cmds.shadingNode('lambert', asShader=True)
    
    file_texture = cmds.shadingNode('file', asTexture=True, name='file_texture')
    cmds.setAttr(file_texture + '.fileTextureName', noise_path, type="string")
    
    place2d_texture = cmds.shadingNode('place2dTexture', asUtility=True, name='place2d_texture')
    
    cmds.connectAttr(place2d_texture + '.outUV', file_texture + '.uvCoord')
    cmds.connectAttr(place2d_texture + '.outUvFilterSize', file_texture + '.uvFilterSize')
    
    cmds.connectAttr(file_texture + '.outColor', lambert_shader + '.color')
    cmds.setAttr(file_texture + '.alphaIsLuminance', 1)
    
    displacement_node = cmds.shadingNode('displacementShader', asShader=True)
    
    cmds.connectAttr(file_texture + '.outAlpha', displacement_node + '.displacement')
    
    shading_group = cmds.sets(renderable=True, noSurfaceShader=True, empty=True)
    
    cmds.connectAttr(lambert_shader + '.outColor', shading_group + '.surfaceShader')
    cmds.connectAttr(displacement_node + '.displacement', shading_group + '.displacementShader')
    
    plane_shape = cmds.listRelatives(plane, shapes=True)[0]
    cmds.sets(plane_shape, edit=True, forceElement=shading_group)
    
    cmds.toggleDisplacement(plane)
    
genPlane()