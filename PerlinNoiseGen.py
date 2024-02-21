import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from opensimplex import OpenSimplex
import math
import platform
import OpenEXR, Imath

# Heightmap size
width = 1024
height = 1024


"""
STEP 1: CREATE A HEIGHTMAP USING PERLIN NOISE
"""


def create_heightmap(width: int, height: int, scale: float = 512.0, seed: int = 42):
    # Create a 2D array of floats
    heightmap = np.zeros((width, height), dtype=float)

    # Create a Perlin noise object
    noise_generator = OpenSimplex(seed=seed)

    # Generate Perlin noise and assign it to the heightmap
    for i in range(height):
        for j in range(width):
            # Generate multiple octaves of noise
            for octave in range(10):  # Adjust the number of octaves as needed
                frequency = 2 ** octave
                amplitude = 1 / frequency
                heightmap[i][j] += (1 - abs(noise_generator.noise2(i / scale * frequency, j / scale * frequency))) * amplitude



    # Normalize the heightmap to the range [0, 1]
    heightmap -= heightmap.min()
    heightmap /= heightmap.max()

    # Invert the heightmap values to create ridges at high values
    heightmap = 1 - heightmap

    return heightmap
    
def save_heightmap_exr(heightmap, filepath):
    # Ensure heightmap is in the range [0, 1]
    heightmap = np.clip(heightmap, 0, 1)

    # Create OpenEXR output file
    exr_header = OpenEXR.Header(heightmap.shape[1], heightmap.shape[0])
    exr_header['compression'] = Imath.Compression(Imath.Compression.NO_COMPRESSION)
    exr_header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    exr_file = OpenEXR.OutputFile(filepath, exr_header)

    # Convert heightmap data to bytes
    exr_data = heightmap.astype(np.float32).tobytes()  # Convert heightmap to 32-bit floating point and convert to bytes

    # Write the heightmap data to the EXR file
    exr_file.writePixels({'R': exr_data})

    # Close the EXR file
    exr_file.close()

heightmap = create_heightmap(width, height, seed=56456)

plt.figure(figsize=(width / 100, height / 100)) 
plt.imshow(heightmap, cmap='gray', interpolation='none')
plt.axis('off')  # Turn off axis

save_heightmap_exr(heightmap, 'C:/Users/tomme/Pictures/heightmap.exr')

# plt.savefig('C:/Users/tomme/Pictures/heightmap.exr', bbox_inches='tight', pad_inches=0)  # Save the image without extra padding
# plt.show()

noise_path = 'C:/Users/tomme/Pictures/heightmap.exr'



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