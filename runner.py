import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as spfft
import scipy.io as sio
import optparse
import concurrent.futures
from numba import jit
from sparseRecovery.solvers import BasisPursuit
from sparseRecovery.solvers import OrthogonalMP
import predictionFunction
from PIL import Image
from sympy import fwht, ifwht
from math import remainder
from processingFunctionsPerChannel import processingFunctionsPerChannel

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
    options                       = get_options()
    operatingColorChannel         = 'rgb'
    subBlockSize                  = 'cow' # Sensing matrix type
    subBlockSize                  = 16    # This could be moved to optParser section
    samplingRate                  = 64   # This could be moved to optParser section
    slidingWindowSize             = 4
    quantizationSlidingWindowSize = 8
    quantizationBit               = 12

    # read original image
    originalImage                 = Image.open(options.fileName, 'r')
    imgWidth, imgHeight           = originalImage.size
    if operatingColorChannel == 'gray':
        originalImage             = originalImage.convert('L')
        originalImageIntObj       = np.array(originalImage)
        loopChannel               = 1
        recoveredSignal           = np.zeros((imgHeight,imgWidth))
    else:
        originalImage             = originalImage.convert('RGB')
        originalImageIntObj       = np.array(originalImage)
        rOriginalImage, gOriginalImage, bOriginalImage = originalImage.split()
        rOriginalImageIntObj      = np.array(rOriginalImage)
        gOriginalImageIntObj      = np.array(gOriginalImage)
        bOriginalImageIntObj      = np.array(bOriginalImage)
        loopChannel               = 3
        recoveredSignal           = np.zeros((imgHeight,imgWidth))
        gRecoveredSignal          = np.zeros((imgHeight,imgWidth))
        recoveredSignal           = np.zeros((imgHeight,imgWidth))
    originalImageIntObj           = np.array(originalImage)

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

    if operatingColorChannel == 'gray':
        recoveredSignal = processingFunctionsPerChannel(imgHeight, imgWidth, subBlockSize, originalImageIntObj, phi, samplingRate, slidingWindowSize, quantizationSlidingWindowSize, quantizationBit)
        plt.imshow(recoveredSignal, cmap = plt.get_cmap('gray'))
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            rRecoveredSignalFuture = executor.submit(processingFunctionsPerChannel, imgHeight, imgWidth, subBlockSize, rOriginalImageIntObj, phi, samplingRate, slidingWindowSize, quantizationSlidingWindowSize, quantizationBit)
            gRecoveredSignalFuture = executor.submit(processingFunctionsPerChannel, imgHeight, imgWidth, subBlockSize, gOriginalImageIntObj, phi, samplingRate, slidingWindowSize, quantizationSlidingWindowSize, quantizationBit)
            bRecoveredSignalFuture = executor.submit(processingFunctionsPerChannel, imgHeight, imgWidth, subBlockSize, bOriginalImageIntObj, phi, samplingRate, slidingWindowSize, quantizationSlidingWindowSize, quantizationBit)
            rRecoveredSignal       = rRecoveredSignalFuture.result()
            gRecoveredSignal       = gRecoveredSignalFuture.result()
            bRecoveredSignal       = bRecoveredSignalFuture.result()
        # Recombine back to RGB image
        arr        = np.array(originalImage)
        arr[:,:,0] = rRecoveredSignal
        arr[:,:,1] = gRecoveredSignal
        arr[:,:,2] = bRecoveredSignal
        recoveredSignal = Image.fromarray(arr)
        plt.imshow(recoveredSignal) 
    plt.show()
    print("Done")