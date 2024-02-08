import maya.cmds as cmds
import random


plane = cmds.polyPlane(width=10, height=10, subdivisionsX=10, subdivisionsY=10)[0]

place2d_texture = cmds.shadingNode('place2dTexture', asUtility=True, name='place2d_texture')



noise_primary = cmds.shadingNode('noise', asShader=True, name='noise_primary')
noise_secondary = cmds.shadingNode('noise', asShader=True, name='noise_secondary')

cmds.connectAttr(place2d_texture + '.outUV', noise_primary + '.uvCoord')
cmds.connectAttr(place2d_texture + '.outUvFilterSize', noise_primary + '.uvFilterSize')
cmds.connectAttr(place2d_texture + '.outUV', noise_secondary + '.uvCoord')
cmds.connectAttr(place2d_texture + '.outUvFilterSize', noise_secondary + '.uvFilterSize')

cmds.setAttr("noise_primary.amplitude", 0.420)
cmds.setAttr("noise_primary.ratio", 0.441)
cmds.setAttr("noise_primary.frequencyRatio", 2.573)
cmds.setAttr("noise_primary.depthMax", 7)
cmds.setAttr("noise_primary.frequency", 3.497)
cmds.setAttr("noise_primary.noiseType", 1)
# peaks
cmds.setAttr("noise_primary.spottyness", 1)


cmds.setAttr("noise_secondary.amplitude", 0.371)
cmds.setAttr("noise_secondary.ratio", 0.343)
cmds.setAttr("noise_secondary.frequencyRatio", 2.510)
cmds.setAttr("noise_secondary.depthMax", 7)
cmds.setAttr("noise_secondary.frequency", 2.797)
cmds.setAttr("noise_secondary.noiseType", 4)

layered_texture_node = cmds.shadingNode('layeredTexture', asShader=True, name='layered_texture_node')
cmds.connectAttr(noise_primary + ".outAlpha", layered_texture_node + ".inputs[0].alpha")
cmds.connectAttr(noise_primary + ".outColor", layered_texture_node + ".inputs[0].color")

cmds.connectAttr(noise_secondary + ".outAlpha", layered_texture_node + ".inputs[1].alpha")
cmds.connectAttr(noise_secondary + ".outColor", layered_texture_node + ".inputs[1].color")

cmds.setAttr("layered_texture_node.inputs[0].blendMode", 6)
cmds.setAttr("layered_texture_node.alphaIsLuminance", 1)


displacement_node = cmds.shadingNode('displacementShader', asShader=True)
cmds.connectAttr(layered_texture_node + '.outAlpha', displacement_node + '.displacement')

cmds.setAttr(displacement_node + ".scale", 3)

#shading group (blue node)
shading_group = cmds.sets(renderable=True, noSurfaceShader=True, empty=True)

lambert_shader = cmds.shadingNode('lambert', asShader=True)

cmds.connectAttr(lambert_shader + '.outColor', shading_group + '.surfaceShader')
cmds.connectAttr(displacement_node + '.displacement', shading_group + '.displacementShader')

# Assign shading_group to plane
plane_shape = cmds.listRelatives(plane, shapes=True)[0]
cmds.sets(plane_shape, edit=True, forceElement=shading_group)
