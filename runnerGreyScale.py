import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
from PIL import Image
import cvxpy as cvx

import optparse

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
    options = get_options()
    subBlockSize = 16 #This could be moved to optParser section
    samplingRate = 64 #This could be moved to optParser section
    # read original image
    originalImage = Image.open(options.fileName, 'r')
    originalImageIntObj = np.asarray(originalImage)
    imgwidth, imgheight = originalImage.size
    y = np.zeros((int(imgwidth/subBlockSize),int(imgheight/subBlockSize),samplingRate))
    
    # sum of two sinusoids
    n = subBlockSize*subBlockSize
    m = samplingRate
    ri = np.random.choice(n, m, replace=False) # random sample of indices
    ri.sort() # sorting not strictly necessary, but convenient for plotting
    # create idct matrix operator
    A = spfft.idct(np.identity(n), norm='ortho', axis=0)
    A = A[ri]
    # do L1 optimization
    vx = cvx.Variable(n)
    objective = cvx.Minimize(cvx.norm(vx, 1))

    for ii, i in enumerate(range(0,imgheight,subBlockSize)):
        for jj, j in enumerate(range(0,imgwidth,subBlockSize)):
            #2D Read out every subblock size in raster scan and do compressed sensing via for loop
            #later can be parallel on multithread
            pulledBlockImage = originalImageIntObj[i:i+subBlockSize,j:j+subBlockSize, 1]
            # Convert from 2D signal to 1D signal through raster scan
            # make a 1-dimensional view of arr
            flat_signal = pulledBlockImage.ravel()
            y[ii, jj] = flat_signal[ri]

    # Cube extraction

    # Intra-Inter prediction

    # DCT Transform

    # Quantization Table

    # Context Adaptive Binary Arithmatic coding (CABAC)

    # Packet formation

    # for x in range(0,int(imgheight/subBlockSize)):
    #     for y in range(0,int(imgwidth/subBlockSize)):
    #     constraints = [A@vx == y]
    #     prob = cvx.Problem(objective, constraints)
    #     result = prob.solve(verbose=False)

    #     # reconstruct signal
    #     x = np.array(vx.value)
    #     x = np.squeeze(x)
    #     sig = spfft.idct(x, norm='ortho', axis=0)
    #     # reform a numpy array of the original shape
    #     reformedSignal = np.asarray(sig).reshape(subBlockSize, subBlockSize)
    #print(reformedSignal)
    #print(signal)
    #plt.imshow(reformedSignal)
    #plt.show()
    #plt.imshow(signal)
    #plt.show()
