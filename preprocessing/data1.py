import numpy
from scipy import signal
import scipy.io as sio
import matplotlib.pyplot as plt

def preproc():
    # load mat file
    input = sio.loadmat('100m.mat')
    x = input['val']
    # transpose x to a column vector
    xt = numpy.transpose(x)
    # change frequency from 360 to 250
    secs = (len(xt) / 360) * 250
    xnf = signal.resample(xt, secs)
    # rescale the signal
    mn = numpy.min(xnf)
    mx = numpy.max(xnf)
    rsc = (xnf - mn) / (mx - mn)
    # split the signal to 10 equal signals
    xfinal = numpy.vsplit(rsc, 10)
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = xfinal
    #just for checking
    print(xfinal)
    plt.plot(a1,'r')
    plt.show()

preproc()
