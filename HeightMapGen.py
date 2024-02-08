import maya.cmds as cmds
import random


plane = cmds.polyPlane(width=10, height=10, subdivisionsX=10, subdivisionsY=10)[0]

noise_primary = cmds.shadingNode('noise', asShader=True, name='noise_primary')
noise_secondary = cmds.shadingNode('noise', asShader=True, name='noise_secondary')

cmds.setAttr("noise_primary.amplitude", 0.420)
cmds.setAttr("noise_primary.ratio", 0.441)
cmds.setAttr("noise_primary.frequencyRatio", 2.573)
cmds.setAttr("noise_primary.depthMax", 7)
cmds.setAttr("noise_primary.frequency", 3.497)
cmds.setAttr("noise_primary.noiseType", 1)


cmds.setAttr("noise_secondary.amplitude", 0.371)
cmds.setAttr("noise_secondary.ratio", 0.343)
cmds.setAttr("noise_secondary.frequencyRatio", 2.510)
cmds.setAttr("noise_secondary.depthMax", 7)
cmds.setAttr("noise_secondary.frequency", 2.797)
cmds.setAttr("noise_secondary.noiseType", 4)
