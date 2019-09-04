import numpy as np
import torch as torch
import json 
# import math as m
# import collections as col
from sympy.utilities.iterables import multiset_permutations

import plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.figure_factory as ff

def rangeList(lst):
    ll = []
    for l in lst:
        ll = ll+list(l) 
#         print(ll)
    return ll
# rangeList([range(0,3),range(6,9)])

def createProbabilityHyperplanes(numDims=1,numObs=20 ):
    lst = []
    
    #generate boundary probabilities
    for i in range(1,numDims+1):
        if numDims >= i:
            e = np.zeros(numDims)
            e[0:i] = np.ones(i)
            for p in multiset_permutations(e/sum(e)):
                lst.append(p)

    #generate random probabilities (must sum to 1)
    num = numObs-len(lst)
    if num > 0:
        r = np.random.rand(num, numDims)
        for i in range(r.shape[0]):
            lst.append(r[i]/sum(r[i]))

    return np.array(lst)[0:numObs] 

def createRmInput(morphIndices=[.1,.35], timeIndices=[.1,.35], sPermutation=[0.,0.,1.], offset=.1):
    '''
    Format Ge input using an list morphology steps, a list of time steps time, and species weights.
    Times in 0 to 1 range (approximately with offset to 1.1 etc.) and species must sum to 1 scaled by time

    time scales species codes so in the output, the sum of species codes will = time

    '''
    lst = []
    for t in timeIndices:
        lst1 = []
        for m in morphIndices:
            lst2 = [t,m]
            for s in sPermutation:
                lst2.append(s*t)
            lst1.append(lst2)
        lst.append(lst1)
    return np.array(lst)

def createRandomGPTargetsFromModel(model, numObs=10,numSpDims=0,num0SpDims=0, numAddPhConCodes=0, gtmOffset=.1,device='cpu'):
    '''    
    ge.shape = [numObs, 2(t,m) + numSpDims + num0SpDims]
    ph.shape = [numObs,model.shape[-1]] i.e. [numObs, model output dimension]
    
    Create a dataset by sampling from a model, 
    num0SpDims adds Ge zero dimensions for use in expanding the  Ge space with unconstrainted mappings
    numPhConCodes adds zeros to the end of Ph coordinates to add emergent context codes
    offset Ge time and morphology dimension to avoid divide by zero errors
    '''
    assert (model.shape[0] == numSpDims+num0SpDims+2), 'First Model dimension {} must equal numSpDims={} plus num0SpDims={} plus time and morph={}'.format(
            model.shape[0],numSpDims,num0SpDims,2)
    
    r = np.random.rand(numObs,2)+gtmOffset
    sp = createProbabilityHyperplanes(numSpDims, numObs)

    lst = []
    for i in range(numObs):
        tm = r[i] #random [time,morph] value
        ss = (sp[i]/sum(sp[i]))*r[i,0] #species codes scaled by time
        tmss = np.concatenate((tm,ss,np.zeros([num0SpDims]))) #time, morphology, scaled species
        lst.append(tmss)
    ge = np.array(lst)
    tgeIn = torch.Tensor(ge).to(device) #np.concatenate((geIn,np.zeros([numObs,numGeZDims])),axis=1) #add Ge zero columns
    tph = model(tgeIn).to(device) 
    s = list(tph.shape[:-1])
    c = torch.zeros(s+[numAddPhConCodes]).to(device)

    ph2 = torch.cat((tph,c),len(c.shape)-1)
    return tgeIn, ph2.to(device)

def createPhRcTargetsFromModelList(tModels, numSamples,numContextDims=0, geRange=1.1):
    '''
    **This has been moved to gy common routines***************
    Inputs
     tModels: [ a list of template models] P = pModel(G)
     numSamples: number of sample training instances
     numContextDims: number of 0 initialized context dimensions added to gSpace
     geRange: range of timestep and morph step values usually 0-1.1 (.1 - 1 is the usable range inside borders)
     
    Output, shape
     g Target [numSamples, numModels+numContextDims]
     p Target [numSamples, pDims(morphSize*3)]
     weights used to interpolate between template model outputs [numSamples, numModels]

    '''  
    numModels = len(tModels) 
    gList = []
    pList = []
    wList = []
    for i in range(numSamples): 
        #for each sample create a random G[timeIndex, morphIndex]
        ge = np.random.random([2]) * geRange
        tge = torch.Tensor(ge)

        #for each model create a random model weight, all weights sum to 1
        r = np.random.random([numModels])
        sm = np.sum(r)
        w =torch.Tensor(r/sm)

        #build weighted model from templates and ge input
        wm = tModels[0](tge)*w[0]
        for i in range(1,numModels):
            wm = wm + tModels[i](tge)*w[i]
        gList.append(tge)
        pList.append(wm)
        wList.append(w)
    g = torch.stack(gList)
    zs = torch.zeros([numSamples,numContextDims])
    gOut = torch.cat((g,torch.zeros([numSamples,numContextDims])),1)
    pOut = torch.stack(pList)
    wOut = torch.stack(wList)

    return gOut, pOut, wOut


def FormatNetworkToBlender(nPetals, xOffset=0, yOffset=5, zOffset=-1, colPoints=5):
    '''
    input (1,7, 14), one petal, 7 curves, 2 offset, 4x coord, 4ycoord, 4zcoord = 14 coordinates
    output (1, 7, 4, 3)) one petal, 7 curves, 4 coordinates of 3 dimensions (x,y,z)
    
    '''
    nP = np.array(nPetals) #convert to array to allow slicing
    pList = []
    ccList = []
    cList = [] #list of x,y,z for curve
    xList = [] #list of x points
    yList = [] #list of y points
    zList = [] #list of y points
    
    numPetals = nP.shape[0]
    numCurves = nP.shape[1]
    for p in range (numPetals):
        nCv = nP[p]
        for i in range (numCurves):       
            xCurve = nCv[i, xOffset: xOffset+colPoints].tolist() #convert to list to get serializable data type
            yCurve = nCv[i, yOffset:yOffset+colPoints].tolist()
            if (zOffset>0):
                zCurve = nCv[i, zOffset:zOffset+colPoints].tolist()

            for ii in range (len(xCurve)):
                if (zOffset>0):
                    cList.append([xCurve[ii],yCurve[ii], zCurve[ii]]) 
                else:
                    cList.append([xCurve[ii],yCurve[ii], 0])

            ccList.append(cList)
            cList=[]
        pList.append(ccList)
        ccList = []
    return np.array(pList)

    
def FormatBlenderToNetwork(bCurve, typeList=[], hasTime=1, hasZ=0, numPoints=5):
    '''
    format curve for network input

    '''
    bCa = np.array(bCurve) #array curve
    nList  = [] #curve list

    inc = (bCa.shape[0])
    for i in range (0, inc):
        xB = bCa[i ,0:numPoints+1, 0] #x is in column 1
        yB = bCa[i ,0:numPoints+1, 1] #y is in column 2
        zB = bCa[i ,0:numPoints+1, 2] #z is in column 3

        if(hasTime): 
            time = [round((i/(inc-1)),3)]
        else:
            time = []  

        cN = time + typeList + np.ndarray.tolist(xB) + np.ndarray.tolist(yB)  
        if(hasZ): 
            cN = cN + np.ndarray.tolist(zB) 
        nList.append(cN)      

    return np.array(nList)


def FormatNetworkToGraphColumns(nCurve, xOffset=0, yOffset=5, colPoints=5):
    nCa = np.array(nCurve)
    cList = [] #list of curves
    xList = [] #list of x points
    yList = [] #list of y points
    
#     op = xOffset
    numCurves = (nCa.shape[0])
    for i in range (0, numCurves):        
        xCurve = nCa[i, xOffset: xOffset+colPoints] 
        yCurve = nCa[i, yOffset:yOffset+colPoints]
        cList.append([xCurve,yCurve])
    return np.array(cList)

# nCurve = [[1,2,3,5,6,7],[2,3,4,5,6,7]]
# xyCurveList= FormatNetworkToGraphColumns(nCurve, xOffset=0, yOffset=3, colPoints=3)
# xyCurveList

def graphCurves(xyCurves, layout=go.Layout(),labels=[], dashRange=range(0,0)):
    traces = []
    for i in range (0, len(xyCurves)):
        xyCa = np.array(xyCurves[i])
        if i < len(labels):
            label = labels[i]
        else:
            label = str(i)
            
        #create a line/spline for each curve   
        sc = go.Scatter(
            x =  xyCa[0]
            ,y = xyCa[1]
            ,name = label
            ,line = dict(shape='spline')  
            )
        #add dash attributes if within dashRange
        if i in dashRange:
            sc.line['dash'] = '2px,6px'
            sc.marker['symbol'] = 'star'

        traces.append(sc)
            
    return go.Figure(data=traces, layout=layout)

def formatForGraphG(eDict,xDim,yDim):
    '''
    Return a list of x and y triples for [back,current,forward] 
    
    shape [numpoints,xy,bcf] (i.e. (7,2,3))
    b(xy), c(xy) and f(xy) for each row (point)
    
    xDim,yDim are the dimensions used in the graph
    '''
    lst = []
    gg = []
    dims = int(eDict.shape[1]/3) #number of input dimensions
    if xDim >= dims or yDim >= dims:
        return np.array(gg)
    
    numCurves = eDict.shape[0]
    for i in range(numCurves):
        hhB = eDict[:,[xDim, yDim]]
        hh0 = eDict[:,[xDim+dims, yDim+dims]]
        hhF = eDict[:,[xDim+dims*2, yDim+dims*2]]
        
        gg.append([[hhB[i,0], hh0[i,0], hhF[i,0]], [hhB[i,1], hh0[i,1], hhF[i,1]]])

    return np.array(gg)

#formatForGraphG(explore['ge_Est'][0:7],0,1) #shape (7, 2, 3)

def FormatNetworkToGraphRows(nCurve, xOffset=0, yOffset=1, rowPoints=2):
    '''
    format curve for 

    '''
    netCa = np.array(nCurve)
    cList  = [] #list of curves
    xCurve = []
    yCurve = []
    inc= 0
    for curve in netCa:
        xCurve.append(curve[xOffset])
        yCurve.append(curve[yOffset])
        inc = inc+1

        if (inc >= rowPoints):
            cList.append([xCurve,yCurve])            
            xCurve = []
            yCurve =[]
            inc = 0
            
    if len(xCurve)>0:
        cList.append([xCurve,yCurve])  #add leftover to last curve          

    return np.array(cList)

