import numpy as np  
def intraPrediction(averageFrame, intraPredictionBuffer, slidingWindowSize, i, j):
    # no prediction for the first block as always
    # allocate memeory for paramters below
    leftTop                                                 = 0
    top                                                     = np.zeros((slidingWindowSize*2, 1))
    left                                                    = np.zeros((slidingWindowSize*2, 1))
    predictionTemplate                                      = np.zeros((slidingWindowSize, slidingWindowSize))
    intraPredictionBufferWidth, intraPredictionBufferHeight = intraPredictionBuffer.shape
    if(i == 0 and j == 0):
        # this stage will immediatly return empty template
        return predictionTemplate
    else:
        if(((i-1) > 0) and (j-1)> 0):
            # obtain value from bottom right of intraPredictionBuffer
            leftTop = np.array(intraPredictionBuffer[(i-1):(i-1)+slidingWindowSize, (j-1):(j-1)+slidingWindowSize])[3,3]
        else:
            leftTop = 0
    if(((i-1) <= 0)):
        top = np.zeros((slidingWindowSize*2, 1))
    elif(((i-1) > 0) and ((j+1) <= intraPredictionBufferWidth)):
        top = [np.array(intraPredictionBuffer[(i-1):(i-1)+slidingWindowSize, j:j+slidingWindowSize])[3,0],
               np.array(intraPredictionBuffer[(i-1):(i-1)+slidingWindowSize, j:j+slidingWindowSize])[3,1],
               np.array(intraPredictionBuffer[(i-1):(i-1)+slidingWindowSize, j:j+slidingWindowSize])[3,2],
               np.array(intraPredictionBuffer[(i-1):(i-1)+slidingWindowSize, j:j+slidingWindowSize])[3,3],
               np.array(intraPredictionBuffer[(i-1):(i-1)+slidingWindowSize, (j+1):(j+1)+slidingWindowSize])[3,0],
               np.array(intraPredictionBuffer[(i-1):(i-1)+slidingWindowSize, (j+1):(j+1)+slidingWindowSize])[3,1],
               np.array(intraPredictionBuffer[(i-1):(i-1)+slidingWindowSize, (j+1):(j+1)+slidingWindowSize])[3,2],
               np.array(intraPredictionBuffer[(i-1):(i-1)+slidingWindowSize, (j+1):(j+1)+slidingWindowSize])[3,3]]
    else:
        top = [np.array(intraPredictionBuffer[(i-1):(i-1)+slidingWindowSize, j:j+slidingWindowSize])[3,0],
                    np.array(intraPredictionBuffer[(i-1):(i-1)+slidingWindowSize, j:j+slidingWindowSize])[3,1],
                    np.array(intraPredictionBuffer[(i-1):(i-1)+slidingWindowSize, j:j+slidingWindowSize])[3,2],
                    np.array(intraPredictionBuffer[(i-1):(i-1)+slidingWindowSize, j:j+slidingWindowSize])[3,3],
                    0,0,0,0]

    if(((j-1) <= 0)):
        left  = np.zeros((slidingWindowSize*2, 1))
    elif(((i+1) <= intraPredictionBufferHeight) and ((j-1) > 0)):
        left = [np.array(intraPredictionBuffer[i:i+slidingWindowSize, (j-1):(j-1)+slidingWindowSize])[3,0],
                     np.array(intraPredictionBuffer[i:i+slidingWindowSize, (j-1):(j-1)+slidingWindowSize])[3,1],
                     np.array(intraPredictionBuffer[i:i+slidingWindowSize, (j-1):(j-1)+slidingWindowSize])[3,2],
                     np.array(intraPredictionBuffer[i:i+slidingWindowSize, (j-1):(j-1)+slidingWindowSize])[3,3],
                     np.array(intraPredictionBuffer[(i+1):(i+1)+slidingWindowSize, (j-1):(j-1)+slidingWindowSize])[3,0],
                     np.array(intraPredictionBuffer[(i+1):(i+1)+slidingWindowSize, (j-1):(j-1)+slidingWindowSize])[3,1],
                     np.array(intraPredictionBuffer[(i+1):(i+1)+slidingWindowSize, (j-1):(j-1)+slidingWindowSize])[3,2],
                     np.array(intraPredictionBuffer[(i+1):(i+1)+slidingWindowSize, (j-1):(j-1)+slidingWindowSize])[3,3]]
    else:
        left = [np.array(intraPredictionBuffer[i:i+slidingWindowSize, (j-1):(j-1)+slidingWindowSize])[3,0],
                     np.array(intraPredictionBuffer[i:i+slidingWindowSize, (j-1):(j-1)+slidingWindowSize])[3,1],
                     np.array(intraPredictionBuffer[i:i+slidingWindowSize, (j-1):(j-1)+slidingWindowSize])[3,2],
                     np.array(intraPredictionBuffer[i:i+slidingWindowSize, (j-1):(j-1)+slidingWindowSize])[3,3],
                     0, 0, 0, 0]

def verticalReplication(top, slidingWindowSize):
    verticalReplicationOutput = np.zeros((slidingWindowSize, slidingWindowSize))
    for i in range(0, slidingWindowSize):
        for j in range(0, slidingWindowSize):
            verticalReplicationOutput[i,j] = top[i]
    return verticalReplicationOutput

def horizonatalReplication(left, slidingWindowSize):
    horizonatalReplicationOutput = np.zeros((slidingWindowSize, slidingWindowSize))
    for i in range(0, slidingWindowSize):
        for j in range(0, slidingWindowSize):
            horizonatalReplicationOutput[i,j] = left[i]
    return horizonatalReplicationOutput

def meanDC (left, top, slidingWindowSize):
    meanDCOut = np.zeros((slidingWindowSize, slidingWindowSize))
    for i in range(0, slidingWindowSize):
        for j in range(0, slidingWindowSize):
            meanDCOut[i,j] = np.round(np.mean(left[0:slidingWindowSize] + top[0:slidingWindowSize]))
    return meanDCOut

def diagonalDownLeft(top, slidingWindowSize):
    diagonalDownLeftOut = np.zeros((slidingWindowSize, slidingWindowSize))
    a = (top[1] + 2*top[2] + top[3] + 2) / 4
    b = (top[2] + 2*top[3] + top[4] + 2) / 4
    c = (top[3] + 2*top[4] + top[5] + 2) / 4
    d = (top[4] + 2*top[5] + top[6] + 2) / 4
    e = (top[5] + 2*top[6] + top[7] + 2) / 4
    f = (top[6] + 2*top[7] + top[8] + 2) / 4
    g = (top[7] + 3*top[8]          + 2) / 4

    diagonalDownLeftOut[0,0] = a
    diagonalDownLeftOut[0,1] = b
    diagonalDownLeftOut[0,2] = c
    diagonalDownLeftOut[0,3] = d
    diagonalDownLeftOut[1,0] = b
    diagonalDownLeftOut[1,1] = c
    diagonalDownLeftOut[1,2] = d
    diagonalDownLeftOut[1,3] = e
    diagonalDownLeftOut[2,0] = c
    diagonalDownLeftOut[2,1] = d
    diagonalDownLeftOut[2,2] = e
    diagonalDownLeftOut[2,3] = f
    diagonalDownLeftOut[3,0] = d
    diagonalDownLeftOut[3,1] = e
    diagonalDownLeftOut[3,2] = f
    diagonalDownLeftOut[3,3] = g

    return diagonalDownLeftOut

def diagonalDownRight(left, top, leftTop, slidingWindowSize):
    diagonalDownRightOut = np.zeros((slidingWindowSize, slidingWindowSize))
    a = (left[4] + 2*top[3]  + left[2] + 2) / 4
    b = (left[3] + 2*left[2] + left[1] + 2) / 4
    c = (left[2] + 2*left[1] + leftTop + 2) / 4
    d = (left[1] + 2*leftTop + top[1]  + 2) / 4
    e = (leftTop + 2*top[1]  + top[2]  + 2) / 4
    f = (top[1]  + 2*top[2]  + top[3]  + 2) / 4
    g = (top[1]  + 2*top[3]  + top[4]  + 2) / 4

    diagonalDownRightOut[0,0] = d
    diagonalDownRightOut[0,1] = e
    diagonalDownRightOut[0,2] = f
    diagonalDownRightOut[0,3] = g
    diagonalDownRightOut[1,0] = c
    diagonalDownRightOut[1,1] = d
    diagonalDownRightOut[1,2] = e
    diagonalDownRightOut[1,3] = f
    diagonalDownRightOut[2,0] = b
    diagonalDownRightOut[2,1] = c
    diagonalDownRightOut[2,2] = d
    diagonalDownRightOut[2,3] = e
    diagonalDownRightOut[3,0] = a
    diagonalDownRightOut[3,1] = b
    diagonalDownRightOut[3,2] = c
    diagonalDownRightOut[3,3] = b

    return diagonalDownRightOut

def verticalRight(left, top, leftTop, slidingWindowSize):
    verticalRightOut = np.zeros((slidingWindowSize, slidingWindowSize))
    a = (leftTop + top[1]    + 1) / 2
    b = (top[1]  + top[2]    + 1) / 2
    c = (top[2]  + top[3]    + 1) / 2
    d = (top[3]  + top[4]    + 1) / 2
    e = (left[1] + 2*leftTop + top[1] + 2) / 4
    f = (leftTop + 2*top[1]  + top[2] + 2) / 4
    g = (top[1]  + 2*top[2]  + top[3] + 2) / 4
    h = (top[2]  + 2*top[3]  + top[4] + 2) / 4
    i = (leftTop + 2*left[1] + left[2] + 2) / 4
    j = (left[1] + 2*left[2] + left[3] + 2) / 4

    verticalRightOut[0,0] = a
    verticalRightOut[0,1] = b
    verticalRightOut[0,2] = c
    verticalRightOut[0,3] = d
    verticalRightOut[1,0] = e
    verticalRightOut[1,1] = f
    verticalRightOut[1,2] = g
    verticalRightOut[1,3] = h
    verticalRightOut[2,0] = i
    verticalRightOut[2,1] = a
    verticalRightOut[2,2] = b
    verticalRightOut[2,3] = c
    verticalRightOut[3,0] = j
    verticalRightOut[3,1] = e
    verticalRightOut[3,2] = f
    verticalRightOut[3,3] = g

    return verticalRightOut

def horizontalDown(left, top, leftTop, slidingWindowSize):
    horizontalDownOut = np.zeros((slidingWindowSize, slidingWindowSize))
    a = (leftTop + left[1]   + 1) / 2
    b = (left[1] + 2*leftTop + top[1]  + 2) / 4
    c = (leftTop + 2*top[1]  + top[2]  + 2) / 4
    d = (top[1]  + 2*top[2]  + top[3]  + 2) / 4
    e = (left[1] + left[2]   + 1) / 2
    f = (leftTop + 2*left[1] + left[2] + 2) / 4
    g = (left[2] + left[3]   + 1) / 2
    h = (left[1] + 2*left[2] + left[3] + 2) / 4
    i = (left[3] + left[4]   + 1) / 2
    j = (left[2] + 2*left[3] + left[4] + 2) / 4

    horizontalDownOut[0,0] = a
    horizontalDownOut[0,1] = b
    horizontalDownOut[0,2] = c
    horizontalDownOut[0,3] = e
    horizontalDownOut[1,0] = e
    horizontalDownOut[1,1] = f
    horizontalDownOut[1,2] = a
    horizontalDownOut[1,3] = b
    horizontalDownOut[2,0] = g
    horizontalDownOut[2,1] = h
    horizontalDownOut[2,2] = e
    horizontalDownOut[2,3] = f
    horizontalDownOut[3,0] = i
    horizontalDownOut[3,1] = j
    horizontalDownOut[3,2] = g
    horizontalDownOut[3,3] = h

    return horizontalDownOut

def verticalLeft(top, slidingWindowSize):
    verticalLeftOut = np.zeros((slidingWindowSize, slidingWindowSize))
    a = (top[1] + top[2]            + 1) / 2
    b = (top[2] + top[3]            + 1) / 2
    c = (top[3] + top[4]            + 1) / 2
    d = (top[4] + top[5]            + 1) / 2
    e = (top[5] + top[6]            + 1) / 2
    f = (top[1] + 2*top[2] + top[3] + 2) / 4
    g = (top[2] + 2*top[3] + top[4] + 2) / 4
    h = (top[3] + 2*top[4] + top[5] + 2) / 4
    i = (top[4] + 2*top[5] + top[6] + 2) / 4
    j = (top[5] + 2*top[6] + top[7] + 2) / 4

    verticalLeftOut[0,0] = a
    verticalLeftOut[0,1] = b
    verticalLeftOut[0,2] = d
    verticalLeftOut[0,3] = d
    verticalLeftOut[1,0] = f
    verticalLeftOut[1,1] = g
    verticalLeftOut[1,2] = h
    verticalLeftOut[1,3] = i
    verticalLeftOut[2,0] = b
    verticalLeftOut[2,1] = c
    verticalLeftOut[2,2] = d
    verticalLeftOut[2,3] = e
    verticalLeftOut[3,0] = g
    verticalLeftOut[3,1] = h
    verticalLeftOut[3,2] = i
    verticalLeftOut[3,3] = j

    return verticalLeftOut


def horizontalUp(left, slidingWindowSize):
    horizontalUpOut = np.zeros((slidingWindowSize, slidingWindowSize))
    a = (left[1] + left[2]   + 1) / 2
    b = (left[1] + 2*left[2] + left[3] + 2) / 4
    c = (left[2] + left[3]   + 1) / 2
    d = (left[2] + 2*left[3] + left[4] + 2) / 4
    e = (left[3] + left[4]   + 1) / 2
    f = (left[3] + 3*left[4] + 2) / 4
    g = left[4]

    horizontalUpOut[0,0] = a
    horizontalUpOut[0,1] = b
    horizontalUpOut[0,2] = c
    horizontalUpOut[0,3] = d
    horizontalUpOut[1,0] = c
    horizontalUpOut[1,1] = d
    horizontalUpOut[1,2] = e
    horizontalUpOut[1,3] = f
    horizontalUpOut[2,0] = e
    horizontalUpOut[2,1] = f
    horizontalUpOut[2,2] = g
    horizontalUpOut[2,3] = g
    horizontalUpOut[3,0] = g
    horizontalUpOut[3,1] = g
    horizontalUpOut[3,2] = g
    horizontalUpOut[3,3] = g

    return horizontalUpOut
