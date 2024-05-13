import maya.cmds as cmds
import maya.mel as mel
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from opensimplex import OpenSimplex
import math
import platform
import OpenEXR, Imath
from scipy.ndimage import gaussian_filter
from configparser import ConfigParser


class UI:
    def __init__(self):
        
        window = "terrain_window"
        title = "Terragains"
        size = (720, 540)
        self.plane = None
        
        self.config = ConfigParser()
        self.selected_config = None
        
        if cmds.window(window, exists=True):
            cmds.deleteUI(window, window=True)
            
        #create ui    
        window = cmds.window(window, title=title,widthHeight=size)
        main_layout = cmds.columnLayout(adjustableColumn=True)
        
        #fields
        self.user_file_path_field = cmds.textFieldGrp(label='Operating File Path')
        self.scale_field = cmds.intFieldGrp(numberOfFields=2, label='Dimensions', value1=400, value2=400)
        self.octave_field = cmds.intFieldGrp(numberOfFields=2, label='Base & Detail Octaves', value1=4, value2=10)
        self.seed_field = cmds.intFieldGrp(numberOfFields=1, label='Seed', value1=56456)
        self.noise_scale_field = cmds.floatFieldGrp(numberOfFields=1, label='Noise Scale', value1=1024.0)
        self.disp_scale_field = cmds.floatFieldGrp(numberOfFields=1, label='Height Scale', value1=3.0)
        self.ABS_control_field = cmds.floatFieldGrp(numberOfFields=2, label='ABS control', value1=1.1, value2=1.0)
        self.slope_control_field = cmds.floatFieldGrp(numberOfFields=1, label='Slope', value1=0.25)        
        self.feathering_toggle = cmds.checkBox(label='Feathering', value=True)
        self.feathering_amount = cmds.intFieldGrp(numberOfFields=1, label='Feathering Amount', value1=50)
        self.config_selection = cmds.optionMenu(label='Choose config')
        

        cmds.button(label='Update Path', command=self.update_working_file_path)
        cmds.button(label='Load Config', command=self.load_config_btn_active)
        cmds.button(label='Write Config', command=self.write_config_btn_active)

        cmds.button(label='Generate Terrain', command=self.generate_terrain_btn_active)
        cmds.button(label='Edit Heightmap', command=self.edit_heightmap_btn_active)


            
        cmds.showWindow()  
        

            
    def update_working_file_path(self, *args):
        print("updating_working_file_path")
        self.working_file_path = cmds.textFieldGrp(self.user_file_path_field, query=True, text=True)
        self.config_file_path = self.working_file_path + 'config.ini'
        print("config file path is: " + self.config_file_path)
        self.read_config_file()
    
    def read_config_file(self):
        cmds.optionMenu(self.config_selection, edit=True, q = True)
        self.delete_menu_items() # delete all items if any already exist
        self.config.read([self.config_file_path])
        for section in self.config.sections():
            cmds.menuItem(label=section, parent=self.config_selection) 
            print("added config section" + section)
    
    def delete_menu_items(self, *args):
        config_selection_list = cmds.optionMenu(self.config_selection, q=True, itemListLong=True) # itemListLong returns the children
        if config_selection_list:
            cmds.deleteUI(config_selection_list)
        else:
            return
        
    def load_config_btn_active(self, *args):
        
       # self.config.read('config.ini')
        self.selected_config = cmds.optionMenu(self.config_selection, q=True, value=True)
        
        print("using" + self.selected_config + "from" + self.config_file_path)

        if self.config.has_section(self.selected_config):
            print("updating parameters")
            parameters = {
                'width': self.config.getint(self.selected_config, 'width'),
                'height': self.config.getint(self.selected_config, 'height'),
                'base_octaves': self.config.getint(self.selected_config, 'baseOctaves'),
                'detail_octaves': self.config.getint(self.selected_config, 'detailOctaves'),
                'seed': self.config.getint(self.selected_config, 'seed'),
                'noise_scale': self.config.getfloat(self.selected_config, 'noiseScale'),
                'disp_scale': self.config.getfloat(self.selected_config, 'dispScale'),
                'base_abs': self.config.getfloat(self.selected_config, 'baseABS'),
                'detail_abs': self.config.getfloat(self.selected_config, 'detailABS'),
                'slope_ctrl': self.config.getfloat(self.selected_config, 'slopeCtrl'),
            }
            print("read parameters")

            print("updating ui elements with new parameters")
            # Update UI elements with parameter values
            cmds.intFieldGrp(self.scale_field, edit=True, value1=parameters['width'], value2=parameters['height'])
            cmds.intFieldGrp(self.octave_field, edit=True, value1=parameters['base_octaves'], value2=parameters['detail_octaves'])
            cmds.intFieldGrp(self.seed_field, edit=True, value1=parameters['seed'])
            cmds.floatFieldGrp(self.noise_scale_field, edit=True, value1=parameters['noise_scale'])
            cmds.floatFieldGrp(self.disp_scale_field, edit=True, value1=parameters['disp_scale'])
            cmds.floatFieldGrp(self.ABS_control_field, edit=True, value1=parameters['base_abs'], value2=parameters['detail_abs'])
            cmds.floatFieldGrp(self.slope_control_field, edit=True, value1=parameters['slope_ctrl'])
            print("finished update ui elements")
        else:
            print("self.selected_config section not found in the config file.")




    #  parameters = [width, height, base_octaves, detail_octaves, seed, noise_scale, disp_scale, base_abs, detail_abs, slope_ctrl, is_feathered]

        
    def write_config_btn_active(self, *args):
        print("write_config_btn_active function called")
        config = ConfigParser()
        self.config_file_path = self.working_file_path + 'config.ini'
    
        # Create the file if it doesn't exist yet
        with open(self.config_file_path, 'w') as f:
            pass
    
        # Add default config 'Alps'
        if not config.has_section('Alps'):
            config.add_section('Alps')
    
        # Set options under section 'Alps'
        config.set('Alps', 'width', '720')
        config.set('Alps', 'height', '720')
        config.set('Alps', 'baseOctaves', '4')
        config.set('Alps', 'detailOctaves', '10')
        config.set('Alps', 'seed', '56456')
        config.set('Alps', 'noiseScale', '512.0')
        config.set('Alps', 'dispScale', '3')
        config.set('Alps', 'baseABS', '1')
        config.set('Alps', 'detailABS', '0.6')
        config.set('Alps', 'slopeCtrl', '0.3')
    
        # Write the updated configuration to file
        with open(self.config_file_path, 'w') as f:
            config.write(f)
            print('config written')
            
        

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
        isFeathered = cmds.checkBox(self.feathering_toggle, query=True, value=True) 
        workingFilePath = self.working_file_path
        self.terrain = Terrain(width, height, seed, noiseScale, dispScale, baseABS, detailABS, slopeCtrl, baseOctaves, detailOctaves, isFeathered, workingFilePath)
    
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
        isFeathered = cmds.checkBox(self.feathering_toggle, query=True, value=True)          
        workingFilePath = self.working_file_path
        
        # Update the Terrain instance with the new parameters
        self.terrain.__init__(width, height, seed, noiseScale, dispScale, baseABS, detailABS, slopeCtrl, baseOctaves, detailOctaves, isFeathered, workingFilePath)
        

        
class Terrain:
    def __init__(self, width, height, seed, noiseScale, dispScale, baseABS, detailABS, slopeCtrl, baseOctaves, detailOctaves, isFeathered, workingFilePath):
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
        self.isFeathered = isFeathered
        self.workingFilePath = workingFilePath
        self.heightmap = self.create_heightmap(self.width, self.height, self.seed, self.noiseScale, self.baseABS, self.detailABS, self.slopeCtrl, self.baseOctaves, self.detailOctaves, self.isFeathered)
        self.plane = None # Initialize plane attribute for editing heightmap

        
       # plt.figure(figsize=(width / 100, height / 100)) 
      #  plt.imshow(self.heightmap, cmap='gray', interpolation='none')
       # plt.axis('off')  # Turn off axis
        self.noise_path = workingFilePath + 'heightmap.exr'

        self.write_heightmap_exr(self.heightmap, workingFilePath)
        # plt.writefig('C:/Users/tomme/Pictures/heightmap.exr', bbox_inches='tight', pad_inches=0)  # write the image without extra padding
        # plt.show()

        self.genPlane(self.noise_path, dispScale)
        
    # for stuff outside 
    def create_heightmap(self, width, height, seed, noiseScale, baseABS, detailABS, slopeCtrl, baseOctaves, detailOctaves, isFeathered):
        
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
            heightmap = self.apply_feathering(heightmap)
            
        return heightmap
        
            
    def apply_feathering(self, heightmap):
        # Define feathering width
        feather_width = 100
        #feather_falloff == True
        #feather_falloff_outer == False
        # Create an empty mask
        mask = np.zeros_like(heightmap)
        
       # falloff_curve_values = cmds.getAttr(self.falloff_curve + '.curve')

        # Calculate the step size for feathering
        # feather_step_size = 1 / feather_width
        
        # Normalize falloff curve values to [0, 1]
        #falloff_curve_values -= falloff_curve_values.min()
        #falloff_curve_values /= falloff_curve_values.max()
        
        # Generate progressively brighter squares
        for i in range(1, feather_width + 1):
            #calculate the exponential step size
            
            
                #feather_step_size = (1 - np.exp(-i)) / (1 - np.exp(-feather_width))
                feather_step_size = np.exp(-i / feather_width)          

        
                # Slice the mask array to create a smaller square
                sliced_mask = mask[i:-i, i:-i]
                # Add the feather step size to the center of the sliced mask
                sliced_mask += feather_step_size
            
        # Normalize the mask to ensure the maximum value is 1
        mask /= np.max(mask)

        return heightmap * mask
        
    def apply_feathering_old(self, heightmap):
    # Create a mask for feathering
        mask = np.ones_like(heightmap)
        mask[:feather_width, :] *= np.linspace(0, 1, feather_width)[:, np.newaxis]
        mask[-feather_width:, :] *= np.linspace(1, 0, feather_width)[:, np.newaxis]
        mask[:, :feather_width] *= np.linspace(0, 1, feather_width)[np.newaxis, :]
        mask[:, -feather_width:] *= np.linspace(1, 0, feather_width)[np.newaxis, :]

        mask = gaussian_filter(mask, sigma=sigma)
    
        # Apply the mask to the heightmap
        feathered_heightmap = heightmap * mask
        
    def write_heightmap_exr(self, heightmap, filepath):
        # Ensure heightmap is in the range [0, 1]
        heightmap = np.clip(heightmap, 0, 1)

        # Create OpenEXR output file
        exr_header = OpenEXR.Header(heightmap.shape[1], heightmap.shape[0])
        exr_header['compression'] = Imath.Compression(Imath.Compression.NO_COMPRESSION)
        exr_header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
        exr_file = OpenEXR.OutputFile(filepath + 'heightmap.exr', exr_header)

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