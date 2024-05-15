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
        self.user_file_path_field = cmds.textFieldGrp(label='Working File Path')
        cmds.button(label='Update Path', command=self.update_working_file_path)

        self.scale_field = cmds.intFieldGrp(numberOfFields=2, label='Dimensions', value1=200, value2=200)
        self.octave_field = cmds.intFieldGrp(numberOfFields=2, label='Base & Detail Octaves', value1=4, value2=10)
        self.seed_field = cmds.intFieldGrp(numberOfFields=1, label='Seed', value1=56456)
        self.noise_scale_field = cmds.floatFieldGrp(numberOfFields=1, label='Noise Scale', value1=100.0)
        self.disp_scale_field = cmds.floatFieldGrp(numberOfFields=1, label='Height Scale', value1=3.0)
        self.ABS_control_field = cmds.floatFieldGrp(numberOfFields=2, label='ABS control', value1=1.1, value2=1.0)
        self.slope_control_field = cmds.floatFieldGrp(numberOfFields=1, label='Slope', value1=0.25)        
        self.feathering_toggle = cmds.checkBox(label='Feathering', value=True)
        #self.feathering_amount = cmds.intFieldGrp(numberOfFields=1, label='Feathering Amount', value1=50)
        self.feathering_scale_slider = cmds.floatSliderGrp(label='Feathering Scale', field=True, minValue=1, maxValue=500, fieldMinValue=1, fieldMaxValue=500, value=50, changeCommand=self.update_feathering_scale)
        
        self.terrain_params = {
            'scale': self.scale_field, 
            'base_octaves': self.octave_field,
            'detail_octaves': self.octave_field,
            'seed': self.seed_field,
            'noise_scale': self.noise_scale_field,
            'disp_scale': self.disp_scale_field,
            'base_abs': self.ABS_control_field,
            'detail_abs': self.ABS_control_field,
            'slope_ctrl': self.slope_control_field,
            'is_feathered': self.feathering_toggle,
            'feathering_amount': self.feathering_scale_slider,
            'working_file_path': self.user_file_path_field,
        }
        
        self.config_selection = cmds.optionMenu(label='Choose config')
        cmds.button(label='Load Config', command=self.load_config_btn_active)
        
        self.config_save_name_field = cmds.textFieldGrp(label='New Config Name')
        cmds.button(label='Save New Config', command=self.write_config_btn_active)
        
        cmds.button(label='Generate Terrain', command=self.generate_terrain_btn_active)
        cmds.button(label='Refresh Noise', command=self.edit_noise_btn_active)
        
        

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
                            
            self.terrain_params['scale'] = self.config.getint(self.selected_config, 'width')
            self.terrain_params['scale'] = self.config.getint(self.selected_config, 'height')
            self.terrain_params['base_octaves'] = self.config.getint(self.selected_config, 'baseOctaves')
            self.terrain_params['detail_octaves'] = self.config.getint(self.selected_config, 'detailOctaves')
            self.terrain_params['seed'] = self.config.getint(self.selected_config, 'seed')
            self.terrain_params['noise_scale'] = self.config.getfloat(self.selected_config, 'noiseScale')
            self.terrain_params['disp_scale'] = self.config.getfloat(self.selected_config, 'dispScale')
            self.terrain_params['base_abs'] = self.config.getfloat(self.selected_config, 'baseABS')
            self.terrain_params['detail_abs'] = self.config.getfloat(self.selected_config, 'detailABS')
            self.terrain_params['slope_ctrl'] = self.config.getfloat(self.selected_config, 'slopeCtrl')
            

            print("read parameters")

            print("updating ui elements with new parameters")
            # Update UI elements with parameter values
            cmds.intFieldGrp(self.scale_field, edit=True, value1=self.terrain_params['scale'], value2=self.terrain_params['scale'])
            cmds.intFieldGrp(self.octave_field, edit=True, value1=self.terrain_params['base_octaves'], value2=self.terrain_params['detail_octaves'])
            cmds.intFieldGrp(self.seed_field, edit=True, value1=self.terrain_params['seed'])
            cmds.floatFieldGrp(self.noise_scale_field, edit=True, value1=self.terrain_params['noise_scale'])
            cmds.floatFieldGrp(self.disp_scale_field, edit=True, value1=self.terrain_params['disp_scale'])
            cmds.floatFieldGrp(self.ABS_control_field, edit=True, value1=self.terrain_params['base_abs'], value2=self.terrain_params['detail_abs'])
            cmds.floatFieldGrp(self.slope_control_field, edit=True, value1=self.terrain_params['slope_ctrl'])
            print("finished update ui elements")
        else:
            print("self.selected_config section not found in the config file.")


    #  parameters = [width, height, base_octaves, detail_octaves, seed, noise_scale, disp_scale, base_abs, detail_abs, slope_ctrl, is_feathered]
    def create_default_config_file(self, *args):
        
        config = ConfigParser()
        
        self.update_working_file_path()
        # Create the file if it doesn't exist yet
        with open(self.config_file_path, 'w') as f:
            pass
        
        # Add default config 'Alps'
        if not config.has_section('Alps' *args):
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
        
        
    def write_config_btn_active(self, *args):
        print("write_config_btn_active function called")
        self.update_working_file_path()
        width, height, seed, noiseScale, dispScale, baseABS, detailABS, slopeCtrl, baseOctaves, detailOctaves, isFeathered, workingFilePath = self.getParameters()
        config = ConfigParser()
        self.config_file_path = self.working_file_path + 'config.ini'

        self.new_config_name = cmds.textFieldGrp(self.config_save_name_field, query=True, text=True)
        
        if not config.has_section(self.new_config_name *args):
            config.add_section(self.new_config_name)
            config.set(self.new_config_name, 'width', width)
            config.set(self.new_config_name, 'height', height)
            config.set(self.new_config_name, 'baseOctaves', baseOctaves)
            config.set(self.new_config_name, 'detailOctaves', detailOctaves)
            config.set(self.new_config_name, 'seed', seed)
            config.set(self.new_config_name, 'noiseScale', noiseScale)
            config.set(self.new_config_name, 'dispScale', dispScale)
            config.set(self.new_config_name, 'baseABS', baseABS)
            config.set(self.new_config_name, 'detailABS', detailABS)
            config.set(self.new_config_name, 'slopeCtrl', slopeCtrl)
        
        
        # Write the updated configuration to file
        with open(self.config_file_path, 'w') as f:
            config.write(f)
            print('config written')
            
            
    def getParameters(self, *args):
        self.terrain_params['scale'] = cmds.intFieldGrp(self.scale_field, query=True, value=True)
        self.terrain_params['scale'] = cmds.intFieldGrp(self.scale_field, query=True, value=True)[1]
        self.terrain_params['base_octaves'] = cmds.intFieldGrp(self.octave_field, query=True, value=True)[0]
        self.terrain_params['detail_octaves'] = cmds.intFieldGrp(self.octave_field, query=True, value=True)[1]
        self.terrain_params['seed'] = cmds.intFieldGrp(self.seed_field, query=True, value=True)[0]
        self.terrain_params['noise_scale'] = cmds.floatFieldGrp(self.noise_scale_field, query=True, value=True)[0]
        self.terrain_params['disp_scale'] = cmds.floatFieldGrp(self.disp_scale_field, query=True, value=True)[0]
        self.terrain_params['base_abs'] = cmds.floatFieldGrp(self.ABS_control_field, query=True, value=True)[0]
        self.terrain_params['detail_abs'] = cmds.floatFieldGrp(self.ABS_control_field, query=True, value=True)[1]    
        self.terrain_params['slope_ctrl'] = cmds.floatFieldGrp(self.slope_control_field, query=True, value=True)[0]
        self.terrain_params['is_feathered'] = cmds.checkBox(self.feathering_toggle, query=True, value=True) 
        self.terrain_params['feathering_amount'] = int(cmds.floatSliderGrp(self.feathering_scale_slider, query=True, value=True))
        self.terrain_params['working_file_path'] = self.working_file_path

    def generate_terrain_btn_active(self, *args):
        self.getParameters()
        self.terrain = Terrain(self.terrain_params)
    
    def edit_noise_btn_active(self, *args):
               
        self.terrain.delPlane()   
        self.getParameters()          
        # Update the Terrain instance with the new parameters
        self.terrain = Terrain(self.terrain_params)
        
    def update_feathering_scale(self, *args):
        self.terrain_params['feathering_amount'] = int(cmds.floatSliderGrp(self.feathering_scale_slider, query=True, value=True))
        print("feathering scale is: " + str(self.terrain_params['feathering_amount']))
        
        self.terrain.update_feathering(self.terrain.baseHeightmap, self.terrain_params)
        print("feathering updated")     
        #wait for a to elapse before applying feathering, if slider moves again, reset timer
         # Cancel the previous script job if it exists
         
         
         
         
         
         
         
class Terrain:
    def __init__(self, terrain_params):
        self.terrain_params = terrain_params
        self.finalHeightmap = self.create_heightmap()
        self.plane = None # Initialize plane attribute for editing heightmap
        # plt.figure(figsize=(width / 100, height / 100)) 
        # plt.imshow(self.heightmap, cmap='gray', interpolation='none')
        # plt.axis('off')  # Turn off axis
        self.noise_path = self.terrain_params['working_file_path'] + 'heightmap.exr'

        self.write_heightmap_exr(self.noise_path)
        # plt.writefig('C:/Users/tomme/Pictures/heightmap.exr', bbox_inches='tight', pad_inches=0)  # write the image without extra padding
        # plt.show()

        self.genPlane(self.noise_path, self.plane)
        
    # for stuff outside 
    def create_heightmap(self):
        print("create_heightmap called")
        
        # Create a Perlin noise object
        noise_generator = OpenSimplex(seed=self.terrain_params['seed'])
        
        # Create a 2D array of floats
        self.baseHeightmap = np.zeros((self.terrain_params['scale'], self.terrain_params['scale']), dtype=float)

        # Generate Perlin noise and assign it to the heightmap
        for i in range(self.terrain_params['scale']):
            for j in range(self.terrain_params['scale']):
                # Generate multiple octaves of noise
                for octave in range(self.terrain_params['base_octaves']):  # Adjust the number of octaves as needed
                    frequency = 2 ** octave
                    amplitude = 1 / frequency
                    self.baseHeightmap[i][j] += (1 - abs(noise_generator.noise2(i / self.terrain_params['noise_scale'] * frequency, j / self.terrain_params['noise_scale'] * frequency))** self.terrain_params['base_abs']) * amplitude
                    
        
    # heightmap = 1 - heightmaph
        
        # Generate fine detail noise
        detail_noise = np.zeros((self.terrain_params['scale'], self.terrain_params['scale']), dtype=float)
        for i in range(self.terrain_params['scale']):
            for j in range(self.terrain_params['scale']):
                for octave in range(self.terrain_params['base_octaves'], self.terrain_params['detail_octaves']):  # Fine detail octaves
                    frequency = 2 ** octave
                    amplitude = 2 / frequency
                    detail_noise[i][j] += (1 - abs(noise_generator.noise2(i / self.terrain_params['noise_scale'] * frequency, j / self.terrain_params['noise_scale'] * frequency))** self.terrain_params['detail_abs']) * amplitude
        
        

        # Apply detail noise to the base heightmap, with less detail on steeper slopes
        slope_factor = np.arctan(np.gradient(self.baseHeightmap, edge_order=2)[0] ** 2 + np.gradient(self.baseHeightmap, edge_order=2)[1] ** 2) ** self.terrain_params['slope_ctrl']
        slope_factor = np.clip(slope_factor, 0, 1)  # Limit the slope factor to the range [0, 1]
    
        # Normalize the self.baseHeightmap to the range [0, 1]
        self.baseHeightmap -= self.baseHeightmap.min()
        self.baseHeightmap /= self.baseHeightmap.max()
        # Normalize the detail noise
        detail_noise -= detail_noise.min()
        detail_noise /= detail_noise.max()
        
        self.baseHeightmap += detail_noise * slope_factor
                
        # Invert the baseHeightmap values to create ridges at high values
        # baseHeightmap = 1 - baseHeightmap
        
        
        
        if self.terrain_params['is_feathered'] == True:
            feathered_heightmap = self.apply_feathering(self.baseHeightmap, self.terrain_params)
            print("returned final heightmap")
        else:
            self.finalHeightmap = self.baseHeightmap
            print("ERRR returned noise output")
            
        return feathered_heightmap
        
                    
    def apply_feathering(self, baseHeightmap, terrain_params):
        print("apply_feathering called")
                
        self.terrain_params = terrain_params
        self.terrain_params['feathering_amount'] = int(self.terrain_params['feathering_amount'])

        print("feathering amount in feathering method is: " + str(self.terrain_params['feathering_amount']))
        mask = np.zeros_like(self.baseHeightmap)
        
        # Generate progressively brighter squares
        for i in range(1, self.terrain_params['feathering_amount'] + 1):
            # Calculate the exponential step size based on the current iteration and total iterations
                feather_step_size = np.exp(-i / self.terrain_params['feathering_amount'])       
                                
        
                # Slice the mask array to create a smaller square
                sliced_mask = mask[i:-i, i:-i]
                # Add the feather step size to the center of the sliced mask
                sliced_mask += feather_step_size
            
        print('feather step size after calc is: ' + str(feather_step_size))
            
        # Normalize the mask to ensure the maximum value is 1
        mask /= np.max(mask)
        
        return self.baseHeightmap * mask
        
    # def apply_feathering_old(self, heightmap):
    # # Create a mask for feathering
    #     mask = np.ones_like(heightmap)
    #     mask[:feather_width, :] *= np.linspace(0, 1, feather_width)[:, np.newaxis]
    #     mask[-feather_width:, :] *= np.linspace(1, 0, feather_width)[:, np.newaxis]
    #     mask[:, :feather_width] *= np.linspace(0, 1, feather_width)[np.newaxis, :]
    #     mask[:, -feather_width:] *= np.linspace(1, 0, feather_width)[np.newaxis, :]

    #     mask = gaussian_filter(mask, sigma=sigma)
    
    #     # Apply the mask to the heightmap
    #     feathered_heightmap = heightmap * mask
        
    def write_heightmap_exr(self, filepath):
        print("write_heightmap_exr called")
        # Ensure heightmap is in the range [0, 1]
        
        
        self.finalHeightmap = np.clip(self.finalHeightmap, 0, 1)

        # Create OpenEXR output file
        exr_header = OpenEXR.Header(self.finalHeightmap.shape[1], self.finalHeightmap.shape[0])
        exr_header['compression'] = Imath.Compression(Imath.Compression.NO_COMPRESSION)
        exr_header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
        exr_file = OpenEXR.OutputFile(filepath, exr_header)

        # Convert heightmap data to bytes
        exr_data = self.finalHeightmap.astype(np.float32).tobytes()  # Convert heightmap to 32-bit floating point and convert to bytes

        # Write the heightmap data to the EXR file
        exr_file.writePixels({'R': exr_data})

        # Close the EXR file
        exr_file.close()


        


    # Generate plane and assign heightmap to displacement shader
    def genPlane(self, noise_path, plane):
        print("genPlane called")
        self.plane = plane
        self.plane = cmds.polyPlane(width=10, height=10, subdivisionsX=10, subdivisionsY=10, n='Landscape')[0]
        
        
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
        cmds.setAttr(displacement_node + '.scale', self.terrain_params['disp_scale'])

    def update_feathering(self, baseHeightmap, terrain_params):
        print("update_feathering called")
        # Update the feathering amount
        
        print("feathering scale in terrain class is: " + str(self.terrain_params['feathering_amount']))
        # Apply feathering to the heightmap
        self.finalHeightmap = self.apply_feathering(self.baseHeightmap, self.terrain_params)
        
        # Write the feathered heightmap to an EXR file
        self.write_heightmap_exr(self.noise_path) 
        
        self.delPlane()
        
        # Generate a plane and assign the feathered heightmap to the displacement shader
        self.genPlane(self.noise_path, self.plane)


    def delPlane(self):
        print("delPlane called")
        cmds.delete(self.plane)
        

UI()