import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as spfft
import scipy.io as sio
from sparseRecovery.solvers import BasisPursuit
from sparseRecovery.solvers import OrthogonalMP
import predictionFunction
from PIL import Image
from sympy import fwht, ifwht
from math import remainder

import optparse

def idct(x):
    return spfft.idct(x.T, norm='ortho')

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

# default parameter options
def get_options():
    optParser = optparse.OptionParser()
    #optParser.add_option("-a", "--add-file", dest="afile", help="additional file")
    optParser.add_option("-f", dest="fileName", help="the pic that will be processed and compressed")
    #optParser.add_option("-b", dest="macroBlockSize", help="block size")
    #optParser.add_option("-sr", dest="samplingRate", help="sampling rate")
    # optParser.add_option("--suffix", dest="suffix",
    #                      help="suffix for output filenames")
    options, args = optParser.parse_args()
    return options

# this is the main entry point of this script
if __name__ == "__main__":
    print("Parameter initialize")
    options             = get_options()
    subBlockSize        = 'cow' # Sensing matrix type
    subBlockSize        = 16    # This could be moved to optParser section
    samplingRate        = 2   # This could be moved to optParser section
    slidingWindowSize   = 4
    # read origina l image
    originalImage       = Image.open(options.fileName, 'r')
    originalImage       = originalImage.convert('L')
    originalImageIntObj = np.array(originalImage)
    imgWidth, imgHeight = originalImage.size
    undeterminedSignal  = np.zeros((int(imgHeight/subBlockSize),int(imgWidth/subBlockSize),samplingRate))
    recoveredSignal     = np.zeros((imgHeight,imgWidth))

    # compressed sensing parameter setup
    n = subBlockSize*subBlockSize
    m = samplingRate

    # create sensing matrix
    phi = sio.loadmat('cowMatrix.mat')
    phi = (phi['cowMatrix'])
    #phi        = sphadamard.hadamard(n)
    # code to replace all negative value with 0
    phi[phi<0] = 0
    # Slicing nxn to mxn
    phi = np.double(phi[0:m,0:n])

    print("Sparse projection")
    for ii, i in enumerate(range(0,imgHeight,subBlockSize)):
        for jj, j in enumerate(range(0,imgWidth,subBlockSize)):
            #2D Read out every subblock size in raster scan and do compressed sensing via for loop
            #later can be parallel on multithread
            pulledBlockImage = originalImageIntObj[i:i+subBlockSize, j:j+subBlockSize]
            # Convert from 2D signal to 1D signal through raster scan
            # make a 1-dimensional view of arr
            flat_signal = np.double(pulledBlockImage.ravel())
            undeterminedSignal[ii, jj, :] = (phi*flat_signal.T).sum(axis=1)
    # Let begin the compression process
    # First, we need to check whatever sub image size can fit into standard sliding window or not.
    # For instance, 240/8 = 30 but 135/8 = 16.875 so we have to pad until it can fit to 17
    padSubImageHeight = imgHeight/subBlockSize
    padSubImageWidth  = imgWidth/subBlockSize
    while (remainder(padSubImageHeight,slidingWindowSize) != 0): padSubImageHeight = padSubImageHeight + 1
    while (remainder(padSubImageWidth,slidingWindowSize) != 0): padSubImageWidth = padSubImageWidth + 1
    
    # Allocate memory for datacube padSubImageWidth
    dataCube = np.array(np.zeros((int(padSubImageHeight), int(padSubImageWidth),samplingRate)))
    # Copy undeterminedSignal to dataCube space where its already paded.
    # ***There must be more efficient way to do this, it will be optimized later
    for i in range(0,int(imgHeight/subBlockSize)):
        for j in range(0,int(imgWidth/subBlockSize)):
            dataCube[i,j,:] = undeterminedSignal[i, j, :]
    dataCube.tofile("unCompressedCube.bin")
    # We assume that we will process layer by layer, 
    # which will result in multiple subimages based on the volmatric data.
    # Each layer of volmumatric data can be accessed via this parameter
    # dataCube[:, :, i], where i is a layer parameter which should not be larger than m
    # Intra-Inter prediction
    #Firstly,we obtain average data frame from datacube this can be done though simple average function or GMM
    # Allocate memory for averageFrame
    averageFrame = np.array(np.zeros((int(padSubImageHeight), int(padSubImageWidth))))
    for i in range(0,samplingRate): 
        averageFrame = averageFrame + dataCube[:,:, i]
    averageFrame = np.floor(averageFrame/samplingRate)

    # After that we create prediction template by applying intra/inter 
    # where sliding window can be varies but must be equal to slidingWindowSize parameter
    # Allocate memory for datacube padSubImageWidth
    predictionTemplate        = np.array(np.zeros((int(padSubImageHeight), int(padSubImageWidth))))
    intraPredictionBuffer     = np.array(np.zeros((int(padSubImageHeight), int(padSubImageWidth))))
    intraPredictionBufferTemp = np.array(np.zeros((slidingWindowSize, slidingWindowSize)))
    # The first procedure here is to creating prediction template for all layer
    for ii, i in enumerate(range(0,int(padSubImageHeight), slidingWindowSize)):
        for jj, j in enumerate(range(0,int(padSubImageWidth), slidingWindowSize)):
            intraPredictionBufferTemp = predictionFunction.intraPrediction(averageFrame, intraPredictionBuffer, slidingWindowSize, i, j, ii, jj)
            predictionTemplate[ii*slidingWindowSize:(ii*slidingWindowSize)+slidingWindowSize, jj*slidingWindowSize:(jj*slidingWindowSize)+slidingWindowSize] = intraPredictionBufferTemp
            intraPredictionBuffer[ii*slidingWindowSize:(ii*slidingWindowSize)+slidingWindowSize, jj*slidingWindowSize:(jj*slidingWindowSize)+slidingWindowSize] = averageFrame[ii*slidingWindowSize:(ii*slidingWindowSize)+slidingWindowSize, jj*slidingWindowSize:(jj*slidingWindowSize)+slidingWindowSize]
            # Transform coding

            # Quantization

            # Dequantization

            # Transform decoding

    # Allocate memory
    codedDatacube = np.array(np.zeros((int(padSubImageHeight), int(padSubImageWidth),samplingRate)))
    decodedDatacube = np.array(np.zeros((int(padSubImageHeight), int(padSubImageWidth),samplingRate)))
    # The second procedure here is to make a real prediction
    for i in range(0,samplingRate):
        codedDatacube[:, :, i] = dataCube[:,:,i] - predictionTemplate
        # Transform coding

        # Quantization

        # Dequantization

        # Transform decoding

        decodedDatacube[:, :, i] = codedDatacube[:,:,i] + predictionTemplate
    
    codedDatacube.tofile("compressedCube.bin")
    # Context Adaptive Binary Arithmatic coding (CABAC) or just Huffman or Arithmatic coding

    # Packet formation

    # Some network simulation is required here in case rate control has been included in framework

    print("Signal recovery via convex optimization")
    # Signal recovery via convex optimization on software or hardware acceleration
    # L1 optimization setup
    # This will recover until meet n dimensional
    for ii, i in enumerate(range(0,imgHeight,subBlockSize)):
        for jj, j in enumerate(range(0,imgWidth,subBlockSize)):
            # initialize solution
            xr = BasisPursuit(phi, decodedDatacube[ii, jj, :])
            recoveredSignal[i:i+subBlockSize, j:j+subBlockSize] = xr.reshape(subBlockSize, subBlockSize)
    # converting 2d list into 1d
    # using list comprehension
    plt.imshow(recoveredSignal, cmap=plt.get_cmap('gray'))
    plt.show()
    print("Done")