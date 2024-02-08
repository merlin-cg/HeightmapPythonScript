import maya.cmds as cmds
import random


plane = cmds.polyPlane(width=10, height=10, subdivisionsX=10, subdivisionsY=10)[0]

file_texture_noise1 = cmds.shadingNode('file', asShader=True, name='file_texture_noise1')



