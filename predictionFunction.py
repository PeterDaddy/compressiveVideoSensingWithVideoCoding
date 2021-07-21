import numpy as np  
def intraPrediction(averageFrame, intraPredictionBuffer, slidingWindowSize, i, j, ii, jj):
    # no prediction for the first block as always
    # allocate memeory for paramters below
    leftTop                                                 = 0
    top                                                     = np.zeros((slidingWindowSize*2, 1))
    left                                                    = np.zeros((slidingWindowSize*2, 1))
    predictionTemplate                                      = np.zeros((slidingWindowSize, slidingWindowSize))
    intraPredictionBufferHeight, intraPredictionBufferWidth = intraPredictionBuffer.shape
    if(ii == 0 and jj == 0):
        # this stage will immediatly return empty template
        return predictionTemplate
    else:
        if(((ii-1) > 0) and (jj-1) > 0):
            # obtain value from bottom right of intraPredictionBuffer
            leftTop = np.array(intraPredictionBuffer[((ii-1)*slidingWindowSize):((ii-1)*slidingWindowSize)+slidingWindowSize, ((jj-1)*slidingWindowSize):((jj-1)*slidingWindowSize)+slidingWindowSize])[3,3]
        else:
            leftTop = 0

    if(((ii-1) <= 0)):
        top = np.zeros((slidingWindowSize*2, 1))
    elif(((ii-1) > 0) and ((jj+1) < (intraPredictionBufferWidth/slidingWindowSize))):
        top = [np.array(intraPredictionBuffer[((ii-1)*slidingWindowSize):((ii-1)*slidingWindowSize)+slidingWindowSize, ((jj)*slidingWindowSize):((jj)*slidingWindowSize)+slidingWindowSize])[3,0],
               np.array(intraPredictionBuffer[((ii-1)*slidingWindowSize):((ii-1)*slidingWindowSize)+slidingWindowSize, ((jj)*slidingWindowSize):((jj)*slidingWindowSize)+slidingWindowSize])[3,1],
               np.array(intraPredictionBuffer[((ii-1)*slidingWindowSize):((ii-1)*slidingWindowSize)+slidingWindowSize, ((jj)*slidingWindowSize):((jj)*slidingWindowSize)+slidingWindowSize])[3,2],
               np.array(intraPredictionBuffer[((ii-1)*slidingWindowSize):((ii-1)*slidingWindowSize)+slidingWindowSize, ((jj)*slidingWindowSize):((jj)*slidingWindowSize)+slidingWindowSize])[3,3],
               np.array(intraPredictionBuffer[((ii-1)*slidingWindowSize):((ii-1)*slidingWindowSize)+slidingWindowSize, ((jj+1)*slidingWindowSize):((jj+1)*slidingWindowSize)+slidingWindowSize])[3,0],
               np.array(intraPredictionBuffer[((ii-1)*slidingWindowSize):((ii-1)*slidingWindowSize)+slidingWindowSize, ((jj+1)*slidingWindowSize):((jj+1)*slidingWindowSize)+slidingWindowSize])[3,1],
               np.array(intraPredictionBuffer[((ii-1)*slidingWindowSize):((ii-1)*slidingWindowSize)+slidingWindowSize, ((jj+1)*slidingWindowSize):((jj+1)*slidingWindowSize)+slidingWindowSize])[3,2],
               np.array(intraPredictionBuffer[((ii-1)*slidingWindowSize):((ii-1)*slidingWindowSize)+slidingWindowSize, ((jj+1)*slidingWindowSize):((jj+1)*slidingWindowSize)+slidingWindowSize])[3,3]]
    else:
        top = [np.array(intraPredictionBuffer[((ii-1)*slidingWindowSize):((ii-1)*slidingWindowSize)+slidingWindowSize, ((jj)*slidingWindowSize):((jj)*slidingWindowSize)+slidingWindowSize])[3,0],
               np.array(intraPredictionBuffer[((ii-1)*slidingWindowSize):((ii-1)*slidingWindowSize)+slidingWindowSize, ((jj)*slidingWindowSize):((jj)*slidingWindowSize)+slidingWindowSize])[3,1],
               np.array(intraPredictionBuffer[((ii-1)*slidingWindowSize):((ii-1)*slidingWindowSize)+slidingWindowSize, ((jj)*slidingWindowSize):((jj)*slidingWindowSize)+slidingWindowSize])[3,2],
               np.array(intraPredictionBuffer[((ii-1)*slidingWindowSize):((ii-1)*slidingWindowSize)+slidingWindowSize, ((jj)*slidingWindowSize):((jj)*slidingWindowSize)+slidingWindowSize])[3,3],
               0,0,0,0]

    if(((jj-1) <= 0)):
        left = np.zeros((slidingWindowSize*2, 1))
    elif(((ii+1) < (intraPredictionBufferHeight/slidingWindowSize)) and ((jj-1) > 0)):
        left = [np.array(intraPredictionBuffer[((ii)*slidingWindowSize):((ii)*slidingWindowSize)+slidingWindowSize, ((jj-1)*slidingWindowSize):((jj-1)*slidingWindowSize)+slidingWindowSize])[0,3],
                np.array(intraPredictionBuffer[((ii)*slidingWindowSize):((ii)*slidingWindowSize)+slidingWindowSize, ((jj-1)*slidingWindowSize):((jj-1)*slidingWindowSize)+slidingWindowSize])[1,3],
                np.array(intraPredictionBuffer[((ii)*slidingWindowSize):((ii)*slidingWindowSize)+slidingWindowSize, ((jj-1)*slidingWindowSize):((jj-1)*slidingWindowSize)+slidingWindowSize])[2,3],
                np.array(intraPredictionBuffer[((ii)*slidingWindowSize):((ii)*slidingWindowSize)+slidingWindowSize, ((jj-1)*slidingWindowSize):((jj-1)*slidingWindowSize)+slidingWindowSize])[3,3],
                np.array(intraPredictionBuffer[((ii+1)*slidingWindowSize):((ii+1)*slidingWindowSize)+slidingWindowSize, ((jj-1)*slidingWindowSize):((jj-1)*slidingWindowSize)+slidingWindowSize])[0,3],
                np.array(intraPredictionBuffer[((ii+1)*slidingWindowSize):((ii+1)*slidingWindowSize)+slidingWindowSize, ((jj-1)*slidingWindowSize):((jj-1)*slidingWindowSize)+slidingWindowSize])[1,3],
                np.array(intraPredictionBuffer[((ii+1)*slidingWindowSize):((ii+1)*slidingWindowSize)+slidingWindowSize, ((jj-1)*slidingWindowSize):((jj-1)*slidingWindowSize)+slidingWindowSize])[2,3],
                np.array(intraPredictionBuffer[((ii+1)*slidingWindowSize):((ii+1)*slidingWindowSize)+slidingWindowSize, ((jj-1)*slidingWindowSize):((jj-1)*slidingWindowSize)+slidingWindowSize])[3,3]]
    else:
        left = [np.array(intraPredictionBuffer[((ii)*slidingWindowSize):((ii)*slidingWindowSize)+slidingWindowSize, ((jj-1)*slidingWindowSize):((jj-1)*slidingWindowSize)+slidingWindowSize])[0,3],
                np.array(intraPredictionBuffer[((ii)*slidingWindowSize):((ii)*slidingWindowSize)+slidingWindowSize, ((jj-1)*slidingWindowSize):((jj-1)*slidingWindowSize)+slidingWindowSize])[1,3],
                np.array(intraPredictionBuffer[((ii)*slidingWindowSize):((ii)*slidingWindowSize)+slidingWindowSize, ((jj-1)*slidingWindowSize):((jj-1)*slidingWindowSize)+slidingWindowSize])[2,3],
                np.array(intraPredictionBuffer[((ii)*slidingWindowSize):((ii)*slidingWindowSize)+slidingWindowSize, ((jj-1)*slidingWindowSize):((jj-1)*slidingWindowSize)+slidingWindowSize])[3,3],
                0, 0, 0, 0]

    predictionTemplate = sumOfAbsoluteDifference(verticalReplication(top, slidingWindowSize),
                                                 horizonatalReplication(left, slidingWindowSize), 
                                                 meanDC (left, top, slidingWindowSize),
                                                 diagonalDownLeft(top, slidingWindowSize),
                                                 diagonalDownRight(left, top, leftTop, slidingWindowSize), 
                                                 verticalRight(left, top, leftTop, slidingWindowSize), 
                                                 horizontalDown(left, top, leftTop, slidingWindowSize), 
                                                 verticalLeft(top, slidingWindowSize),
                                                 horizontalUp(left, slidingWindowSize),
                                                 averageFrame[i:i+slidingWindowSize, j:j+slidingWindowSize],
                                                 slidingWindowSize)
    return predictionTemplate

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
    a = (top[0] + 2*top[1] + top[2] + 2) / 4
    b = (top[1] + 2*top[2] + top[3] + 2) / 4
    c = (top[2] + 2*top[3] + top[4] + 2) / 4
    d = (top[3] + 2*top[4] + top[5] + 2) / 4
    e = (top[4] + 2*top[5] + top[6] + 2) / 4
    f = (top[5] + 2*top[6] + top[7] + 2) / 4
    g = (top[6] + 3*top[7]          + 2) / 4

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
    a = (left[3] + 2*top[2]  + left[1] + 2) / 4
    b = (left[2] + 2*left[1] + left[0] + 2) / 4
    c = (left[1] + 2*left[0] + leftTop + 2) / 4
    d = (left[0] + 2*leftTop + top[0]  + 2) / 4
    e = (leftTop + 2*top[0]  + top[1]  + 2) / 4
    f = (top[0]  + 2*top[1]  + top[2]  + 2) / 4
    g = (top[0]  + 2*top[2]  + top[3]  + 2) / 4

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
    a = (leftTop + top[0]    + 1) / 2
    b = (top[0]  + top[1]    + 1) / 2
    c = (top[1]  + top[2]    + 1) / 2
    d = (top[2]  + top[3]    + 1) / 2
    e = (left[0] + 2*leftTop + top[0] + 2) / 4
    f = (leftTop + 2*top[0]  + top[1] + 2) / 4
    g = (top[0]  + 2*top[1]  + top[2] + 2) / 4
    h = (top[1]  + 2*top[2]  + top[3] + 2) / 4
    i = (leftTop + 2*left[0] + left[1] + 2) / 4
    j = (left[0] + 2*left[1] + left[2] + 2) / 4

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
    a = (leftTop +   left[0] + 1) / 2
    b = (left[0] + 2*leftTop + top[0]  + 2) / 4
    c = (leftTop + 2*top[0]  + top[1]  + 2) / 4
    d = (top[0]  + 2*top[1]  + top[2]  + 2) / 4
    e = (left[0] +   left[1] + 1) / 2
    f = (leftTop + 2*left[0] + left[1] + 2) / 4
    g = (left[1] +   left[2] + 1) / 2
    h = (left[0] + 2*left[1] + left[2] + 2) / 4
    i = (left[2] +   left[3] + 1) / 2
    j = (left[1] + 2*left[2] + left[3] + 2) / 4

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
    a = (top[0] +   top[1]          + 1) / 2
    b = (top[1] +   top[2]          + 1) / 2
    c = (top[2] +   top[3]          + 1) / 2
    d = (top[3] +   top[4]          + 1) / 2
    e = (top[4] +   top[5]          + 1) / 2
    f = (top[0] + 2*top[1] + top[2] + 2) / 4
    g = (top[1] + 2*top[2] + top[3] + 2) / 4
    h = (top[2] + 2*top[3] + top[4] + 2) / 4
    i = (top[3] + 2*top[4] + top[5] + 2) / 4
    j = (top[4] + 2*top[5] + top[6] + 2) / 4

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
    a = (left[0] +   left[1] + 1) / 2
    b = (left[0] + 2*left[1] + left[2] + 2) / 4
    c = (left[1] +   left[2] + 1) / 2
    d = (left[1] + 2*left[2] + left[3] + 2) / 4
    e = (left[2] +   left[3] + 1) / 2
    f = (left[2] + 3*left[3] + 2) / 4
    g =  left[3]

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

def sumOfAbsoluteDifference(verticalReplication, horizonatalReplication, meanDC, diagonalDownLeft, 
                            diagonalDownRight, verticalRight, horizontalDown, verticalLeft, horizontalUp, 
                            averageFrame, slidingWindowSize):
    SADTable = [round(np.sum(np.sum(np.abs(verticalReplication    - averageFrame)))),
                round(np.sum(np.sum(np.abs(horizonatalReplication - averageFrame)))),
                round(np.sum(np.sum(np.abs(meanDC                 - averageFrame)))),
                round(np.sum(np.sum(np.abs(diagonalDownLeft       - averageFrame)))),
                round(np.sum(np.sum(np.abs(diagonalDownRight      - averageFrame)))),
                round(np.sum(np.sum(np.abs(verticalRight          - averageFrame)))),
                round(np.sum(np.sum(np.abs(horizontalDown         - averageFrame)))),
                round(np.sum(np.sum(np.abs(verticalLeft           - averageFrame)))),
                round(np.sum(np.sum(np.abs(horizontalUp           - averageFrame))))]

    minimumIndexInSAD = SADTable.index(min(SADTable))
    if(minimumIndexInSAD == 0): 
        print('Vertical Replication')
        intraPredictionCandidate = np.floor(verticalReplication)
    elif(minimumIndexInSAD == 1):
        print('Horizonatal Replication')
        intraPredictionCandidate = np.floor(horizonatalReplication)
    elif(minimumIndexInSAD == 2):
        print('Mean/DC')
        intraPredictionCandidate = np.floor(meanDC)
    elif(minimumIndexInSAD == 3):
        print('Diagonal Down-Left')
        intraPredictionCandidate = np.floor(diagonalDownLeft)
    elif(minimumIndexInSAD == 4):
        print('Diagonal Down-Right')
        intraPredictionCandidate = np.floor(diagonalDownRight)
    elif(minimumIndexInSAD == 5):
        print('Vertical Right')
        intraPredictionCandidate = np.floor(verticalRight)
    elif(minimumIndexInSAD == 6):
        print('Horizontal Down')
        intraPredictionCandidate = np.floor(horizontalDown)
    elif(minimumIndexInSAD == 7):
        print('Vertical Left')
        intraPredictionCandidate = np.floor(verticalLeft)
    elif(minimumIndexInSAD == 8):
        print('Horizontal Up')
        intraPredictionCandidate = np.floor(horizontalUp)
    else:
        print('How did you get here?')
        intraPredictionCandidate = np.zeros((slidingWindowSize, slidingWindowSize))

    return intraPredictionCandidate

    #    function candidate_value = inter_prediction(a, b, c, i, j)
    #        % Overlapping searching
    #        % a -> original
    #        % b -> previous frame
    #        % c -> search_area. ex = 4
    #        % i -> x
    #        % j -> y
    #        cell2mat_b = cell2mat(b);
    #        if((i==1) || (i==(size(a,1)) || (j==1) || (j==(size(a,2))))) 
    #            %___Stop working at bouandary zone___
    #            candidate_value = cell2mat_b(i:i+c-1, j:j+c-1);%___Then equal to coordinate___
    #        else
    #            for ii = 1:(c*2)
    #                for jj = 1:(c*2)
    #                    mag_search(ii,jj) = sum(sum(abs(a{i,j}-cell2mat_b(ii:ii+c-1, jj:jj+c-1))));
    #                end
    #            end
    #            [pos_i, pos_j]  = find(min(mag_search(mag_search > 0)) == mag_search(:,:));
    #            candidate_value = cell2mat_b(pos_i:pos_i+c-1, pos_j:pos_j+c-1);
    #        end
    #    end