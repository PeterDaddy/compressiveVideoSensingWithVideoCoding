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
            leftTop     = 0
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
