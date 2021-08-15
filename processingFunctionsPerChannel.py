import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import optparse
import concurrent.futures
import predictionFunction
import entropy
import os

from numba import jit
from basisPursuit.l1eq_pd import l1eq_pd
from sparseRecovery.solvers import BasisPursuit
from sparseRecovery.solvers import OrthogonalMP
from sparseRecovery.solvers import MatchingPursuit
from numpy.core.fromnumeric import compress
from PIL import Image
from sympy import fwht, ifwht
from math import remainder
from scipy.fftpack import dct, idct

# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')  

def processingFunctionsPerChannel(imgHeight, imgWidth, subBlockSize, originalImageIntObj, intraPredictionBuffer, interPredictionBuffer,
                                  phi, theta, samplingRate, slidingWindowSize, quantizationSlidingWindowSize, quantizationBit):
                                  
    undeterminedSignal = np.zeros((int(imgHeight/subBlockSize),int(imgWidth/subBlockSize), samplingRate))
    recoveredSignal    = np.zeros((imgHeight,imgWidth))

    # Let begin the compression process
    # First, we need to check whatever sub image size can fit into standard sliding window or not.
    # For instance, 240/8 = 30 but 135/8 = 16.875 so we have to pad until it can fit to 17
    padSubImageHeight  = imgHeight/subBlockSize
    padSubImageWidth   = imgWidth/subBlockSize
    while (remainder(padSubImageHeight, slidingWindowSize) != 0): padSubImageHeight = padSubImageHeight + 1
    while (remainder(padSubImageWidth, slidingWindowSize) != 0): padSubImageWidth   = padSubImageWidth + 1

    # After that we create prediction template by applying intra/inter
    # where sliding window can be varies but must be equal to slidingWindowSize parameter
    # Allocate memory for datacube padSubImageWidth
    #intraPredictionBuffer     = np.array(np.zeros((int(padSubImageHeight), int(padSubImageWidth))))
    #interPredictionBuffer     = np.array(np.zeros((int(padSubImageHeight), int(padSubImageWidth))))

    predictionTemplate        = np.array(np.zeros((int(padSubImageHeight), int(padSubImageWidth))))
    predictionMaps            = np.array(np.zeros((int(imgHeight), int(imgWidth), 3)))
    predictionModes           = np.array(np.zeros((int(padSubImageHeight), int(padSubImageWidth))))
    intraPredictionBufferTemp = np.array(np.zeros((slidingWindowSize, slidingWindowSize)))
    interPredictionBufferTemp = np.array(np.zeros((slidingWindowSize, slidingWindowSize)))

    # Allocate memory
    codedDatacube             = np.array(np.zeros((int(padSubImageHeight), int(padSubImageWidth), samplingRate)))
    decodedDatacube           = np.array(np.zeros((int(padSubImageHeight), int(padSubImageWidth), samplingRate)))
    dctcodedDatacube          = np.array(np.zeros((int(padSubImageHeight), int(padSubImageWidth), samplingRate)))
    idctcodedDatacube         = np.array(np.zeros((int(padSubImageHeight), int(padSubImageWidth), samplingRate)))
    quantizedcodedDatacube    = np.array(np.zeros((int(padSubImageHeight), int(padSubImageWidth), samplingRate)))
    dequantizedcodedDatacube  = np.array(np.zeros((int(padSubImageHeight), int(padSubImageWidth), samplingRate)))

    uncompressed_entropy      = 0
    compressed_entropy        = 0

    #print("Sparse projection")
    for ii, i in enumerate(range(0,imgHeight,subBlockSize)):
        for jj, j in enumerate(range(0,imgWidth,subBlockSize)):
            #2D Read out every subblock size in raster scan and do compressed sensing via for loop
            #later can be parallel on multithread
            pulledBlockImage = originalImageIntObj[i:i+subBlockSize, j:j+subBlockSize]
            # Convert from 2D signal to 1D signal through raster scan
            # make a 1-dimensional view of arr
            flat_signal = np.double(pulledBlockImage.ravel())
            undeterminedSignal[ii, jj, :] = (phi*flat_signal.T).sum(axis=1)
    
    # Allocate memory for datacube padSubImageWidth
    dataCube = np.array(np.zeros((int(padSubImageHeight), int(padSubImageWidth),samplingRate)))
    # Copy undeterminedSignal to dataCube space where its already paded.
    # ***There must be more efficient way to do this, it will be optimized later
    for i in range(0,int(imgHeight/subBlockSize)):
        for j in range(0,int(imgWidth/subBlockSize)):
            dataCube[i,j,:] = undeterminedSignal[i, j, :]

    # We assume that we will process layer by layer, 
    # which will result in multiple subimages based on the volmatric data.
    # Each layer of volmumatric data can be accessed via this parameter
    # dataCube[:, :, i], where i is a layer parameter which should not be larger than m
    # Intra-Inter prediction
    #Firstly,we obtain average data frame from datacube this can be done though simple average function or GMM
    # Allocate memory for averageFrame
    #print('Started generate prediction template')
    averageFrame = np.array(np.zeros((int(padSubImageHeight), int(padSubImageWidth))))
    for i in range(0,samplingRate):
        averageFrame = averageFrame + dataCube[:,:, i]
    averageFrame = np.floor(averageFrame/samplingRate)

    # The first procedure here is to creating prediction template for all layer
    for ii, i in enumerate(range(0,int(padSubImageHeight), slidingWindowSize)):
        for jj, j in enumerate(range(0,int(padSubImageWidth), slidingWindowSize)):
            intraPredictionBufferTemp = predictionFunction.intraPrediction(averageFrame, intraPredictionBuffer, slidingWindowSize, i, j, ii, jj)
            if(np.sum(np.sum(interPredictionBuffer[ii*slidingWindowSize:(ii*slidingWindowSize)+slidingWindowSize, jj*slidingWindowSize:(jj*slidingWindowSize)+slidingWindowSize])) > 0):
                interPredictionBufferTemp = predictionFunction.interPrediction(averageFrame, interPredictionBuffer, slidingWindowSize, i, j, ii, jj)
            judgeIntra = np.floor(np.linalg.norm(averageFrame[ii*slidingWindowSize:(ii*slidingWindowSize)+slidingWindowSize, jj*slidingWindowSize:(jj*slidingWindowSize)+slidingWindowSize] - intraPredictionBufferTemp))
            judgeInter = np.floor(np.linalg.norm(averageFrame[ii*slidingWindowSize:(ii*slidingWindowSize)+slidingWindowSize, jj*slidingWindowSize:(jj*slidingWindowSize)+slidingWindowSize] - interPredictionBufferTemp))
            #print('Intra prediction Magnitude : ', judgeIntra, 'Inter prediction Magnitude : ', judgeInter )
            if(judgeIntra < judgeInter):
                predictionTemplate[ii*slidingWindowSize:(ii*slidingWindowSize)+slidingWindowSize, jj*slidingWindowSize:(jj*slidingWindowSize)+slidingWindowSize] = intraPredictionBufferTemp
            else:
                predictionTemplate[ii*slidingWindowSize:(ii*slidingWindowSize)+slidingWindowSize, jj*slidingWindowSize:(jj*slidingWindowSize)+slidingWindowSize] = interPredictionBufferTemp
            # Store as usual
            intraPredictionBuffer[ii*slidingWindowSize:(ii*slidingWindowSize)+slidingWindowSize, jj*slidingWindowSize:(jj*slidingWindowSize)+slidingWindowSize] = averageFrame[ii*slidingWindowSize:(ii*slidingWindowSize)+slidingWindowSize, jj*slidingWindowSize:(jj*slidingWindowSize)+slidingWindowSize]
            interPredictionBuffer[ii*slidingWindowSize:(ii*slidingWindowSize)+slidingWindowSize, jj*slidingWindowSize:(jj*slidingWindowSize)+slidingWindowSize] = averageFrame[ii*slidingWindowSize:(ii*slidingWindowSize)+slidingWindowSize, jj*slidingWindowSize:(jj*slidingWindowSize)+slidingWindowSize] 
    #plt.imshow(predictionTemplate, cmap=plt.get_cmap('gray'))
    #plt.show()
    #plt.pause(0.0001)
    #print('Compressing...')
    # The second procedure here is to make a real prediction
    # Alright, generally quantization 2x2 and 4x4 are good in transform coding performance but worst in quality
    # Next, 16x16 is not good since the coefficient will be spreaded in large area 
    # So, we will use 8x8 follow with standard quantization in JPEG/JPEG2000 or AV2
    for ii, i in enumerate(range(0,int(padSubImageHeight), quantizationSlidingWindowSize)):
        for jj, j in enumerate(range(0,int(padSubImageWidth), quantizationSlidingWindowSize)):
            for k in range(0, samplingRate):
                codedDatacube[ii*quantizationSlidingWindowSize:(ii*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, jj*quantizationSlidingWindowSize:(jj*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, k] = dataCube[ii*quantizationSlidingWindowSize:(ii*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, jj*quantizationSlidingWindowSize:(jj*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, k] - predictionTemplate[ii*quantizationSlidingWindowSize:(ii*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, jj*quantizationSlidingWindowSize:(jj*quantizationSlidingWindowSize)+quantizationSlidingWindowSize]
                # Transform coding
                dctcodedDatacube[ii*quantizationSlidingWindowSize:(ii*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, jj*quantizationSlidingWindowSize:(jj*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, k] = np.floor(dct2(codedDatacube[ii*quantizationSlidingWindowSize:(ii*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, jj*quantizationSlidingWindowSize:(jj*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, k]))
                # Quantization table for intra
                if(predictionModes[ii, jj] == 0):
                    quantizationTable = [[8, 8, 8, 8, 8, 8, 8, 8],
                                         [8, 8, 8, 8, 8, 8, 8, 8],
                                         [8, 8, 8, 8, 8, 8, 8, 8],
                                         [8, 8, 8, 8, 8, 8, 8, 8],
                                         [8, 8, 8, 8, 8, 8, 8, 8],
                                         [8, 8, 8, 8, 8, 8, 8, 8],
                                         [8, 8, 8, 8, 8, 8, 8, 8],
                                         [8, 8, 8, 8, 8, 8, 8, 8]]
                # Quantization table for inter
                else:
                    # Quantization
                    quantizationTable = [[16, 16, 16, 16, 16, 16, 16, 16],
                                         [16, 16, 16, 16, 16, 16, 16, 16],
                                         [16, 16, 16, 16, 16, 16, 16, 16],
                                         [16, 16, 16, 16, 16, 16, 16, 16],
                                         [16, 16, 16, 16, 16, 16, 16, 16],
                                         [16, 16, 16, 16, 16, 16, 16, 16],
                                         [16, 16, 16, 16, 16, 16, 16, 16],
                                         [16, 16, 16, 16, 16, 16, 16, 16]]
                quantizedcodedDatacube[ii*quantizationSlidingWindowSize:(ii*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, jj*quantizationSlidingWindowSize:(jj*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, k] = np.floor((dctcodedDatacube[ii*quantizationSlidingWindowSize:(ii*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, jj*quantizationSlidingWindowSize:(jj*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, k])/quantizationTable)
                #print('Coded : ',codedDatacube[ii*quantizationSlidingWindowSize:(ii*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, jj*quantizationSlidingWindowSize:(jj*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, k])
                #print('DCT : ',dctcodedDatacube[ii*quantizationSlidingWindowSize:(ii*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, jj*quantizationSlidingWindowSize:(jj*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, k])
                # Dequantization
                dequantizedcodedDatacube[ii*quantizationSlidingWindowSize:(ii*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, jj*quantizationSlidingWindowSize:(jj*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, k] = np.floor((quantizedcodedDatacube[ii*quantizationSlidingWindowSize:(ii*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, jj*quantizationSlidingWindowSize:(jj*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, k])*quantizationTable)
                # Transform decoding
                idctcodedDatacube[ii*quantizationSlidingWindowSize:(ii*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, jj*quantizationSlidingWindowSize:(jj*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, k] = np.floor(idct2(dequantizedcodedDatacube[ii*quantizationSlidingWindowSize:(ii*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, jj*quantizationSlidingWindowSize:(jj*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, k]))
                # Decode intra/inter
                decodedDatacube[ii*quantizationSlidingWindowSize:(ii*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, jj*quantizationSlidingWindowSize:(jj*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, k] = idctcodedDatacube[ii*quantizationSlidingWindowSize:(ii*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, jj*quantizationSlidingWindowSize:(jj*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, k] + predictionTemplate[ii*quantizationSlidingWindowSize:(ii*quantizationSlidingWindowSize)+quantizationSlidingWindowSize, jj*quantizationSlidingWindowSize:(jj*quantizationSlidingWindowSize)+quantizationSlidingWindowSize]
    #plt.imshow(quantizedcodedDatacube[:,:,1], cmap=plt.get_cmap('gray'))
    #plt.show()
    #plt.pause(0.0001)
    # Context Adaptive Binary Arithmatic coding (CABAC) or just Huffman or Arithmatic coding

    # Packet formation

    # Some network simulation is required here in case rate control has been included in framework
    
    #print("Entropy measuring")
    # for i in range(0,int(imgHeight/subBlockSize)):
    #     for j in range(0,int(imgWidth/subBlockSize)):
    #         # Entropy
    #         stat_info = os.stat(dataCube[i,j,:])
    #         # get size of file in bytes
    #         size = stat_info.st_size
    #         print(size)
    #         #uncompressed_entropy = uncompressed_entropy + (-np.sum(marg0*np.log2(marg0)))
    #         #marg1 = (quantizedcodedDatacube[ii, jj, :]/quantizedcodedDatacube[ii, jj, :].sum(axis=0))
    #         #compressed_entropy = compressed_entropy + (-np.sum(marg1*np.log2(marg1)))
    # print('Entropy of uncompressed cube: ', uncompressed_entropy/samplingRate, ' / Entropy of compresed cube: ', compressed_entropy/samplingRate)
    
    #print("Signal recovery via convex optimization")
    # Signal recovery via convex optimization on software or hardware acceleration
    # L1 optimization setup
    # This will recover until meet n dimensional
    # for ii, i in enumerate(range(0,imgHeight,subBlockSize)):
    #     for jj, j in enumerate(range(0,imgWidth,subBlockSize*8)):
    #         # initialize solution
    #         with concurrent.futures.ThreadPoolExecutor() as executor:
    #             jj = jj*8
    #             future_0  = executor.submit(l1eq_pd, (phi.T*decodedDatacube[ii, jj+0, :]).sum(axis=1), phi, [], decodedDatacube [ii, jj+0, :])
    #             future_1  = executor.submit(l1eq_pd, (phi.T*decodedDatacube[ii, jj+1, :]).sum(axis=1), phi, [], decodedDatacube [ii, jj+1, :])
    #             future_2  = executor.submit(l1eq_pd, (phi.T*decodedDatacube[ii, jj+2, :]).sum(axis=1), phi, [], decodedDatacube [ii, jj+2, :])
    #             future_3  = executor.submit(l1eq_pd, (phi.T*decodedDatacube[ii, jj+3, :]).sum(axis=1), phi, [], decodedDatacube [ii, jj+3, :])
    #             future_4  = executor.submit(l1eq_pd, (phi.T*decodedDatacube[ii, jj+4, :]).sum(axis=1), phi, [], decodedDatacube [ii, jj+4, :])
    #             future_5  = executor.submit(l1eq_pd, (phi.T*decodedDatacube[ii, jj+5, :]).sum(axis=1), phi, [], decodedDatacube [ii, jj+5, :])
    #             future_6  = executor.submit(l1eq_pd, (phi.T*decodedDatacube[ii, jj+6, :]).sum(axis=1), phi, [], decodedDatacube [ii, jj+6, :])
    #             future_7  = executor.submit(l1eq_pd, (phi.T*decodedDatacube[ii, jj+7, :]).sum(axis=1), phi, [], decodedDatacube [ii, jj+7, :])

    #             recoveredSignal[i:i+subBlockSize, j+(subBlockSize*0):(j+(subBlockSize*0))+subBlockSize] = future_0.result().reshape(subBlockSize, subBlockSize)
    #             recoveredSignal[i:i+subBlockSize, j+(subBlockSize*1):(j+(subBlockSize*1))+subBlockSize] = future_1.result().reshape(subBlockSize, subBlockSize)
    #             recoveredSignal[i:i+subBlockSize, j+(subBlockSize*2):(j+(subBlockSize*2))+subBlockSize] = future_2.result().reshape(subBlockSize, subBlockSize)
    #             recoveredSignal[i:i+subBlockSize, j+(subBlockSize*3):(j+(subBlockSize*3))+subBlockSize] = future_3.result().reshape(subBlockSize, subBlockSize)
    #             recoveredSignal[i:i+subBlockSize, j+(subBlockSize*4):(j+(subBlockSize*4))+subBlockSize] = future_4.result().reshape(subBlockSize, subBlockSize)
    #             recoveredSignal[i:i+subBlockSize, j+(subBlockSize*5):(j+(subBlockSize*5))+subBlockSize] = future_5.result().reshape(subBlockSize, subBlockSize)
    #             recoveredSignal[i:i+subBlockSize, j+(subBlockSize*6):(j+(subBlockSize*6))+subBlockSize] = future_6.result().reshape(subBlockSize, subBlockSize)
    #             recoveredSignal[i:i+subBlockSize, j+(subBlockSize*7):(j+(subBlockSize*7))+subBlockSize] = future_7.result().reshape(subBlockSize, subBlockSize)
    #         #plt.imshow(recoveredSignal)
    #         #plt.draw()
    #         #plt.pause(0.0001)
    #         #plt.clf()
    return recoveredSignal