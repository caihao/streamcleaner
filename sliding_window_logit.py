import numba
import numpy

@numba.jit(numba.float64(numba.float64[:],numba.float64[:]))
def innerprod(a, b):
        acc = 0
        for i in range(len(a)):
            acc = acc + a[i] * b[i]
        return acc

@numba.jit(numba.float64[:](numba.float64[:],numba.float64[:]))
def same_convolution_float_1d(ap1, ap2): 
        n1= len(ap1)
        n2= len(ap2)

        n_left = n2 // 2
        n_right = n2 - n_left - 1

        ret = numpy.zeros(n1, numpy.float64)

        idx = 0
        inc = 1

        for i in range(n2 - n_left, n2):
            ret[idx] = innerprod(ap1[:i], ap2[-i:])
            idx += inc

        for i in range(n1 - n2 + 1):
            ret[idx] = innerprod(ap1[i : i + n2], ap2)
            idx += inc

        for i in range(n2 - 1, n2 - 1 - n_right, -1):
            ret[idx] = innerprod(ap1[-i:], ap2[:i])
            idx += inc
        return ret

@numba.jit(numba.float64[:](numba.float64[:],numba.int64))
def float_zero_pad_1d(array, pad_width):

    padded = numpy.zeros(array.size+pad_width*2, dtype=numpy.float64)
    # Copy old array into correct space
    padded[pad_width:-pad_width] = array[:]
    padded[:pad_width] = numpy.median(padded[pad_width:pad_width*2])
    padded[-pad_width:] = numpy.median(padded[-pad_width*2:-pad_width])

    return padded


@numba.jit(numba.float64[:](numba.float64[:],numba.int64))
def moving_average(x, w):
    return same_convolution_float_1d(x, numpy.ones(w,numpy.float64)) / w

@numba.jit(numba.float64[:](numba.float64[:],numba.int64))
def smoothpadded(data: numpy.ndarray,n:int):
  return moving_average(float_zero_pad_1d(data, n*2),n)[n*2: -n*2]

@numba.jit()
def generate_true_logistic(points:int):
    fprint = numpy.linspace(0.0,1.0,points)
    fprint[1:-1]  /= 1 - fprint[1:-1]
    fprint[1:-1]  = numpy.log(fprint[1:-1])
    fprint[-1] = ((2*fprint[-2])  - fprint[-3]) 
    fprint[0] = -fprint[-1]
    fprint = (fprint - fprint[0])/fprint[-1] #normalize between 0 and 1
    return fprint


  
@numba.jit()
def find_noise_factor(data: numpy.ndarray,logit_size: int,smooth_size:int):
  if logit_size %2 == 0: #if true, is even size
    pad_before = logit_size//2
    pad_after = logit_size//2
  else:
    pad_before = logit_size//2
    pad_after = logit_size//2+1

  logit_three = generate_true_logistic(logit_size)   
  product1 = numpy.zeros(data.size,dtype=numpy.float64)
  product2 = numpy.zeros(data.size,dtype=numpy.float64)

  for i in range(pad_before,data.size-pad_after):
    temp = data[i-pad_before:i+pad_after]
    positions = numpy.argsort(temp)
    indx = numpy.argwhere(positions==pad_before)[0]
    temp = numpy.sort(temp)
    temp = (temp - temp[0])/temp[-1] ##normalize to 0,1
    a =  numpy.corrcoef(temp,logit_three)[0,1]
    b = numpy.abs(temp[indx] - logit_three[indx])
    c = a*b
    product1[i] = c[0]
    product2[i] = 1 - a

  product1[:pad_before] = product1[pad_before:pad_before*2]
  product1[-pad_after:] = numpy.flip(product1[-pad_after*2:-pad_after])
  product2[:pad_before] = product2[pad_before:pad_before*2]
  product2[-pad_after:] = numpy.flip(product2[-pad_after*2:-pad_after])

  R = smoothpadded(product1,smooth_size)
  T = smoothpadded(product2,smooth_size)
  good = (R*T)*16
  bad = (R/T)/16
  r = bad - good
  r[r<0] = 0
  return r

#logit_size: int,smooth_size:int - logit size should be an odd number close to 1/4th of rate. smooth_size should be an odd number close to 4x rate.
#rate should be the sampling rate / some quantity. For 48000 I chose 100. 
#this lends? time resolution? 10ms frequency resolution? 800? idk
#time to process is close to 0.61x - 1 second of audio in 0.61 seconds, using numba so it by no means catches up with our stft algo
#however- the noise reduction can be said to be generally applicable to all forms of signal, and since it is a very simple algorithm(
#in terms of how much information it considers) then it could even be integrated into fpga.
R = find_noise_factor(measurements,121,1921)

def denoise(data: numpy.ndarray):
  initial = data.copy()
  noise =  find_noise_factor(data,121,1921)
  initialnoiseenergy = initial * noise
  newresult = initial - initialnoiseenergy
  
  return newresult
result = denoise(data)




