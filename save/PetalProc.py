'''
### Used to read Template files from NN into Blender
petals = readMeshdata(filePath)

### Used to import a Petal or move an existing petal to new vertices locations
petals = readMeshdata(filePath)
petalVerts = petals[2][0]
moveMesh(petalVerts)

'''
import bpy
import bmesh
import json
import numpy as np
import sys

from bpy_extras.object_utils import object_data_add
from mathutils import Vector
from contextlib import contextmanager

from math import radians
from mathutils import Matrix

# ------------------------------------------------------------------------------
# BMesh Context manager
# ------------------------------------------------------------------------------
@contextmanager
def bmesh_from_obj(obj, mode):
    """Context manager to auto-manage bmesh regardless of mode."""

    if mode == 'EDIT_MESH':
        bm = bmesh.from_edit_mesh(obj.data)
        #print('cm1',type(bm),obj.data)
    else:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
    yield bm

    bm.normal_update()

    if mode == 'EDIT_MESH':
        bmesh.update_edit_mesh(obj.data)
    else:
        bm.to_mesh(obj.data)

    bm.free()

def readTemplateData(filePath):
    '''
    Reads json template file (time/petals, column/morph steps, row/cross points, coordinates [x,y,z])
    Outputs an array [6,7,7,3]
    '''
    with open(filePath, 'r') as fp:
        cList =json.load(fp)
    return np.array(cList)  
   
def convertTemplateToVertFaces(tArray,scale=1):
    '''
    Inputs an petal grid array (column/morph steps, row/cross points, coordinates [x,y,z]) i.e. [6,7,7,3]
    Outputs a list of template petals each with a list of Blender Vertices and Faces
    Scaled * 10
    ta= gyb.readTemplateData(filePath)
    p = gyb.convertTemplateToVertFaces(ta)
    print('first petal first vector: ',p[0][0][0])
    print('first petal row of faces: \n',p[0][1][0])
    '''
    numPetals = tArray.shape[0]
    numColumns = tArray.shape[1]
    numRows = tArray.shape[2]    
    faces = []
    petals = [] 
    for p in range(numPetals):
        verts = []
        faces = []
        petal = tArray[p]
        for i in range(numColumns):
            facesR = []
            for ii in range(numRows):
                x = petal[i,ii,0]*scale
                y = petal[i,ii,1]*scale
                z = petal[i,ii,2]*scale
                v = Vector((x,y,z))         
                verts.append(v)
                if i > 0 and ii > 0:         
                    facesR.append([numRows*(i-1)+ii-1, numRows*(i-1)+ii+0, numRows*(i)+ii+0, numRows*(i)+ii-1])
            if i > 0:      
                faces.append(facesR)  
        petals.append((verts, np.array(faces)))
    return petals    

def addMeshPetal(petal, SubSurf=False, Mirror=False):
    '''
    take petal from convertTemplateToVertFaces and creates a mesh and object
    '''
    (verts,faceArray) = petal
    faces = faceArray.reshape([-1,4]).tolist()
    mesh = bpy.data.meshes.new(name='Petal')
    mesh.from_pydata(verts, [], faces)
    #    # useful for development when the mesh may be invalid.
    #    # mesh.validate(verbose=True)
    object_data_add(bpy.context, mesh, operator=None)
    
    if Mirror:
        bpy.ops.object.modifier_add(type='MIRROR')
        bpy.context.object.modifiers["Mirror"].use_x = False
        bpy.context.object.modifiers["Mirror"].use_y = True
    if SubSurf:
        subsurf = obj.modifiers.new(name='SubSurf', type='SUBSURF')
        subsurf.levels = 2
        subsurf.render_levels = 3

def moveMesh(petalVerts):
    bm = bmesh.from_edit_mesh(bpy.context.object.data)
    if hasattr(bm.verts, "ensure_lookup_table"): 
        bm.verts.ensure_lookup_table()
        # only if you need to   :
        # bm.edges.ensure_lookup_table()   
        # bm.faces.ensure_lookup_table()
                    
    for i in range(len(petalVerts)):
        vc = bm.verts[i].co #old position
        pc = petalVerts[i]  #new position
        vc.x = pc.x
        vc.y = pc.y
        vc.z = pc.z

    bpy.context.object.select = True

def moveMeshX(petalVerts):
    '''
    get the mesh of the current object, and change its vert locations
    '''
    try:
        with bmesh_from_obj(bpy.context.object, bpy.context.mode) as bm:
            if isinstance(bm,bytes):
                print('t',type(bm))
                return None

            if hasattr(bm.verts, "ensure_lookup_table"): 
                bm.verts.ensure_lookup_table()
                # only if you need to   :
                # bm.edges.ensure_lookup_table()   
                # bm.faces.ensure_lookup_table()
                            
            for i in range(len(petalVerts)):
                vc = bm.verts[i].co #old position
                pc = petalVerts[i]  #new position
                vc.x = pc.x
                vc.y = pc.y
                vc.z = pc.z

        bpy.context.object.select = True
    except OSError as err:
        print("OS error: {0}".format(err))
        #raise
    except ValueError:
        print("Value Error.")
    except AttributeError as error:
        print('error:', error)
    except:
        print("Unexpected eror:", sys.exc_info()[0])
        #raise
              
def addMods(obj=bpy.context.object):
    '''
        be sure that a mesh object is in object mode
        add an empty if necessary and 
    '''
    if not obj:
        print('No object selected')
        return {'FINISHED'}
    elif not obj.type =='MESH':
        print('No mesh selected')
        return {'FINISHED'}
    if obj.mode == 'EDIT':
        obj.mode_set ( mode = 'OBJECT' )
        print('changed to object mode')
              
    #get reference to selected object and add empty if necessary
    
    #smooth, add empty, add subsurf, add displace
    obj.shade_smooth()
    if not ('Empty' in bpy.data.objects):
        bpy.ops.object.empty_add(type='PLAIN_AXES', view_align=False, location=(.0,.0,.0))       
    if not ('SubSurf' in obj.modifiers):
        subsurf = obj.modifiers.new(name='SubSurf', type='SUBSURF')
        subsurf.levels = 2
        subsurf.render_levels = 3
#     if not ('Displace' in obj.modifiers):
#         displace = obj.modifiers.new(name='Displace', type='DISPLACE')
#         displace.texture_coords = 'OBJECT'
#         displace.texture_coords_object = bpy.data.objects["Empty"]
#         displace.strength = 0.025

def duplicatRotate(numPetals=4):        
     #duplcate linked, rotate 
    ang = 360/numPetals
    for i in range(0,numPetals-1):  
        bpy.ops.object.duplicate_move_linked(
            OBJECT_OT_duplicate={"linked":True, "mode":'TRANSLATION'},        
            TRANSFORM_OT_translate={"value":(0,0,0),  
        "constraint_orientation":'LOCAL', "constraint_axis":(False,False,False)}
        )  
        bpy.ops.transform.rotate(value=radians(ang), axis=(0, 0, 1), 
        constraint_axis=(False,False,False), constraint_orientation='LOCAL'
        )

    #    bpy.ops.transform.resize(value=(0.8, 0.8, 0.8), constraint_orientation='LOCAL' 
    #    )

def writeSelectedEdgesToJason(filePath):
    od = bpy.context.object.data
    bm = bmesh.from_edit_mesh(od)
    lst = []
    llst = []
    buf = []
    i=0
    print('start')
    for elem in bm.select_history:
        #print('i,: ',i,elem.verts)
        if not isinstance(elem, bmesh.types.BMEdge):
            continue #not an edge so skip
        v00= elem.verts[0].co
        v11 = elem.verts[1].co
        buf=[]

        lng = len(lst)
        #print('i,lng,v00,v11: ',i,lng,v00,v11)
        if i==0:
            i=i #skip loop
            #i= i+1
        #first element is link       
        elif lng==0 and (v0 == v00):
            lst.append((v1.x, v1.y, v1.z))
            lst.append((v0.x, v0.y, v0.z))
            lst.append((v11.x, v11.y, v11.z))
        elif lng==0 and (v0 == v11):
            lst.append((v1.x, v1.y, v1.z))
            lst.append((v0.x, v0.y, v0.z))
            lst.append((v00.x, v00.y, v00.z))
        #second element is link
        elif lng==0 and (v1 == v00 ):
            lst.append((v0.x, v0.y, v0.z))
            lst.append((v1.x, v1.y, v1.z))
            lst.append((v11.x, v11.y, v11.z))
        elif lng==0 and (v1 == v11 ):
            lst.append((v0.x, v0.y, v0.z))
            lst.append((v1.x, v1.y, v1.z))
            lst.append((v00.x, v00.y, v00.z))
        elif lst[lng-1]==(v00.x, v00.y, v00.z): #v0 =previous v1
            lst.append((v11.x, v11.y, v11.z))
        elif lst[lng-1]==(v11.x, v11.y, v11.z): #v1 =previous v1
            lst.append((v00.x, v00.y, v00.z))
        else:
            llst.append(lst) #add row
            lst=[] #start new row
            buf = [v00,v11]
        #print('shape: ',np.array(lst).shape)
        i = i+1
        v0 = elem.verts[0].co
        v1 = elem.verts[1].co
        
    if len(buf) > 0:
        lst.append((buf[1].x, buf[1].y, buf[1].z))
        lst.append((buf[0].x, buf[0].y, buf[0].z))   
        
    if len(lst) > 0:
        llst.append(lst)

    with open(filePath, 'w') as fp:
        json.dump(llst,fp) 

    return np.array(llst)