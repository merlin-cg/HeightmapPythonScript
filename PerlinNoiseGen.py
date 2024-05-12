import maya.cmds as cmds
import maya.mel as mel
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from opensimplex import OpenSimplex
import math
import platform
import OpenEXR, Imath



class UI:
    def __init__(self):
        
        window = "terrain_window"
        title = "Terragen"
        size = (720, 540)
        self.plane = None
        
        if cmds.window(window, exists=True):
            cmds.deleteUI(window, window=True)
            
        #creat ui    
        window = cmds.window(window, title=title,widthHeight=size)
        main_layout = cmds.columnLayout(adjustableColumn=True)
        
        #fields
        self.scale_field = cmds.intFieldGrp(numberOfFields=2, label='Dimensions', value1=400, value2=400)
        self.octave_field = cmds.intFieldGrp(numberOfFields=2, label='Base & Detail Octaves', value1=4, value2=10)
        self.seed_field = cmds.intFieldGrp(numberOfFields=1, label='Seed', value1=56456)
        self.noise_scale_field = cmds.floatFieldGrp(numberOfFields=1, label='Noise Scale', value1=1024.0)
        self.disp_scale_field = cmds.floatFieldGrp(numberOfFields=1, label='Height Scale', value1=3.0)
        self.ABS_control_field = cmds.floatFieldGrp(numberOfFields=2, label='ABS control', value1=1.1, value2=1.0)
        self.slope_control_field = cmds.floatFieldGrp(numberOfFields=1, label='Slope', value1=0.25)        
        self.feathering_toggle = cmds.checkBox(label='Feathering')
        
        
        cmds.button (label='Generate Terrain', command=self.generate_terrain_btn_active)
        cmds.button(label='Edit Heightmap', command=self.edit_heightmap_btn_active)
                
        cmds.showWindow()
        
    def generate_terrain_btn_active(self, *args):
        width = cmds.intFieldGrp(self.scale_field, query=True, value=True)[0]
        height = cmds.intFieldGrp(self.scale_field, query=True, value=True)[1]
        baseOctaves = cmds.intFieldGrp(self.octave_field, query=True, value=True)[0]
        detailOctaves = cmds.intFieldGrp(self.octave_field, query=True, value=True)[1]
        seed = cmds.intFieldGrp(self.seed_field, query=True, value=True)[0]
        noiseScale = cmds.floatFieldGrp(self.noise_scale_field, query=True, value=True)[0]
        dispScale = cmds.floatFieldGrp(self.disp_scale_field, query=True, value=True)[0]
        baseABS = cmds.floatFieldGrp(self.ABS_control_field, query=True, value=True)[0]
        detailABS = cmds.floatFieldGrp(self.ABS_control_field, query=True, value=True)[1]    
        slopeCtrl = cmds.floatFieldGrp(self.slope_control_field, query=True, value=True)[0]
        isFeathered = cmds.checkBox(self.feathering_toggle, query=True, value=True)[0]
        
        self.terrain = Terrain(width, height, seed, noiseScale, dispScale, baseABS, detailABS, slopeCtrl, baseOctaves, detailOctaves, isFeathered)
        
    def edit_heightmap_btn_active(self, *args):
               
        self.terrain.delPlane()
        
        # Get the parameters from the UI
        width = cmds.intFieldGrp(self.scale_field, query=True, value=True)[0]
        height = cmds.intFieldGrp(self.scale_field, query=True, value=True)[1]
        baseOctaves = cmds.intFieldGrp(self.octave_field, query=True, value=True)[0]
        detailOctaves = cmds.intFieldGrp(self.octave_field, query=True, value=True)[1]
        seed = cmds.intFieldGrp(self.seed_field, query=True, value=True)[0]
        noiseScale = cmds.floatFieldGrp(self.noise_scale_field, query=True, value=True)[0]
        dispScale = cmds.floatFieldGrp(self.disp_scale_field, query=True, value=True)[0]
        baseABS = cmds.floatFieldGrp(self.ABS_control_field, query=True, value=True)[0]
        detailABS = cmds.floatFieldGrp(self.ABS_control_field, query=True, value=True)[1]
        slopeCtrl = cmds.floatFieldGrp(self.slope_control_field, query=True, value=True)[0]
        isFeathered = cmds.checkBox(self.feathering_toggle, query=True, value=True)[0]


        # Update the Terrain instance with the new parameters
        self.terrain.__init__(width, height, seed, noiseScale, dispScale, baseABS, detailABS, slopeCtrl, baseOctaves, detailOctaves, isFeathered)
        
        
class Terrain:
    def __init__(self, width, height, seed, noiseScale, dispScale, baseABS, detailABS, slopeCtrl, baseOctaves, detailOctaves, isFeathered):
        self.width = width
        self.height = height
        self.seed = seed
        self.noiseScale = noiseScale
        self.dispScale = dispScale
        self.baseABS = baseABS
        self.detailABS = detailABS
        self.slopeCtrl = slopeCtrl
        self.baseOctaves = baseOctaves
        self.detailOctaves = detailOctaves
        self.heightmap = self.create_heightmap(self.width, self.height, self.seed, self.noiseScale, self.baseABS, self.detailABS, self.slopeCtrl, self.baseOctaves, self.detailOctaves, self.isFeathered)
        self.plane = None # Initialize plane attribute for editing heightmap
        self.isFeathered = isFeathered
        
       # plt.figure(figsize=(width / 100, height / 100)) 
      #  plt.imshow(self.heightmap, cmap='gray', interpolation='none')
       # plt.axis('off')  # Turn off axis

        self.save_heightmap_exr(self.heightmap, '/home/s5609424/Pictures/heightmap.exr')
        # plt.savefig('C:/Users/tomme/Pictures/heightmap.exr', bbox_inches='tight', pad_inches=0)  # Save the image without extra padding
        # plt.show()

        self.noise_path = '/home/s5609424/Pictures/heightmap.exr'
        self.genPlane(self.noise_path, dispScale)
        
    # for stuff outside 
    def create_heightmap(self, width, height, seed, noiseScale, baseABS, detailABS, slopeCtrl, baseOctaves, detailOctaves):
        
        # Create a Perlin noise object
        noise_generator = OpenSimplex(seed=self.seed)
        
        # Create a 2D array of floats
        heightmap = np.zeros((width, height), dtype=float)
        # Generate Perlin noise and assign it to the heightmap
        for i in range(height):
            for j in range(width):
                # Generate multiple octaves of noise
                for octave in range(self.baseOctaves):  # Adjust the number of octaves as needed
                    frequency = 2 ** octave
                    amplitude = 1 / frequency
                    heightmap[i][j] += (1 - abs(noise_generator.noise2(i / self.noiseScale * frequency, j / self.noiseScale * frequency))** baseABS) * amplitude
                    
        
    # heightmap = 1 - heightmaph
        
        # Generate fine detail noise
        detail_noise = np.zeros((width, height), dtype=float)
        for i in range(height):
            for j in range(width):
                for octave in range(self.baseOctaves, self.detailOctaves):  # Fine detail octaves
                    frequency = 2 ** octave
                    amplitude = 2 / frequency
                    detail_noise[i][j] += (1 - abs(noise_generator.noise2(i / self.noiseScale * frequency, j / self.noiseScale * frequency))** detailABS) * amplitude
        
        

        # Apply detail noise to the base heightmap, with less detail on steeper slopes
        slope_factor = np.arctan(np.gradient(heightmap, edge_order=2)[0] ** 2 + np.gradient(heightmap, edge_order=2)[1] ** 2) ** slopeCtrl
        slope_factor = np.clip(slope_factor, 0, 1)  # Limit the slope factor to the range [0, 1]
    
        # Normalize the heightmap to the range [0, 1]
        heightmap -= heightmap.min()
        heightmap /= heightmap.max() 
        # Normalize the detail noise
        detail_noise -= detail_noise.min()
        detail_noise /= detail_noise.max()
        
        heightmap += detail_noise * slope_factor
                
        # Invert the heightmap values to create ridges at high values
        # heightmap = 1 - heightmap
        
        if isFeathered == True:
            self.apply_feathering(heightmap)
        return heightmap

    def apply_feathering(self, heightmap, sigma=10):
    # Create a Gaussian mask for feathering
    mask = np.zeros_like(heightmap)
    mask[:5, :] = np.linspace(1, 0, 5)  # Feathering down from 1 to 0 over the first 5 rows
    mask[-5:, :] = np.linspace(0, 1, 5)  # Feathering up from 0 to 1 over the last 5 rows
    mask[:, :5] *= np.linspace(1, 0, 5)[:, np.newaxis]  # Feathering from 1 to 0 over the first 5 columns
    mask[:, -5:] *= np.linspace(0, 1, 5)[:, np.newaxis]  # Feathering from 0 to 1 over the last 5 columns

    # Apply Gaussian blur to the mask for smooth feathering
    mask = gaussian_filter(mask, sigma=sigma)

    # Apply the mask to the heightmap
    heightmap = heightmap * mask

    return heightmap
    

        
    def save_heightmap_exr(self, heightmap, filepath):
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



    # Generate plane and assign heightmap to displacement shader
    def genPlane(self, noise_path, dispScale):

        self.plane = cmds.polyPlane(width=10, height=10, subdivisionsX=10, subdivisionsY=10)[0]
        
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
        
        plane_shape = cmds.listRelatives(self.plane, shapes=True)[0]
        cmds.sets(plane_shape, edit=True, forceElement=shading_group)
        
        cmds.toggleDisplacement(self.plane)
        cmds.setAttr(displacement_node + '.scale', dispScale)

    def delPlane(self):
        cmds.delete(self.plane)
        

UI()    