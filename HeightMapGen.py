import maya.cmds as cmds
import random


plane = cmds.polyPlane(width=10, height=10, subdivisionsX=10, subdivisionsY=10)[0]

noise_primary = cmds.shadingNode('noise', asShader=True, name='noise_primary')
noise_secondary = cmds.shadingNode('noise', asShader=True, name='noise_secondary')

