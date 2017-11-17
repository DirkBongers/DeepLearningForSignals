# dora's version with some small modifications
# - fileName and targetFrequency as parameters
# - actual frequency is not hardcoded, but extracted from .info file

import numpy
from scipy import signal
import scipy.io as sio
import matplotlib.pyplot as plt

def preproc(fileName, targetFrequency=250):
	# load mat file
	input = sio.loadmat(fileName + '.mat')
	x = input['val']
	# transpose x to a column vector
	xt = numpy.transpose(x)
	# change frequency
	secs = (len(xt) / getActualFrequency(fileName)) * targetFrequency
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
	
def getActualFrequency(fileName):
	info = open(fileName + '.info','r')
	text = info.read()
	start = 'Sampling frequency: '
	end = 'Hz'
	return int((text.split(start))[1].split(end)[0])

preproc('100m')