import numpy as np
from scipy.stats import invgamma

def kalman(measurements: np.ndarray):

  # Choose prior hyperparameters for inverse-gamma distribution- are these correct?
  alpha = 2
  beta = 2

  # Compute the differences between consecutive measurements
  diffs = np.diff(measurements)

  # Compute the sample mean and sample variance of the differences
  diff_mean = np.mean(diffs)
  diff_var = np.var(diffs)

  # Compute the posterior hyperparameters for inverse-gamma distribution
  post_alpha = alpha + len(diffs)/2
  post_beta = beta + 0.5*np.sum((diffs-diff_mean)**2)

  # Sample from the posterior distribution of R
  R_samples = invgamma.rvs(post_alpha, scale=post_beta, size=1000)

  # Use the posterior mean of R as the initial estimate
  R = np.mean(R_samples)

  # Set up the mathematical model
  delta_t = 1  # time step
  F = np.array([[1, delta_t], [0, 1]])  # state transition matrix
  H = np.array([[1, 0]])  # observation matrix
  Q = np.array([[0.1, 0], [0, 0.01]])  # process noise covariance matrix

  # Initialize the state variables and error covariance matrix
  x_0 = np.array([measurements[0], 0])  # initial state estimate
  P_0 = np.eye(2)  # initial error covariance matrix

  # Initialize the state estimate and error covariance matrix
  x_est = x_0
  P_est = P_0

  # Initialize the measurement noise covariance matrix R
  R_est = np.mean(R_samples)

  x_result = np.zeros_like(measurements)
  i = 0
  # Iterate over the measurements to update the state estimate
  for z in measurements:
    # Predict the state using the state transition matrix and the previous state estimate
    x_pred = np.dot(F, x_est)
    P_pred = np.dot(F, np.dot(P_est, F.T)) + Q
    
    # Calculate the Kalman gain matrix using Numpy
    K = np.dot(P_pred, H.T) / (np.dot(H, np.dot(P_pred, H.T)) + R_est)
    
    # Update the state estimate using the Kalman gain matrix and the measurement
    x_est = x_pred + np.dot(K, z - np.dot(H, x_pred))
    P_est = np.dot(np.eye(2) - np.dot(K, H), P_pred)
    
    # Update the measurement noise covariance matrix R
    r_error = z - np.dot(H, x_pred)
    R_est = np.dot(r_error.T, r_error) / 1  # Using a simple average
    
    # Store the denoised signal estimate
    x_result[i] = x_est[0]
    i = i+1
  return x_result



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

@numba.njit(numba.float64(numba.float64[:]))
def man(arr):
    med = numpy.nanmedian(arr[numpy.nonzero(arr)])
    return numpy.nanmedian(numpy.abs(arr - med))

@numba.njit(numba.float64(numba.float64[:]))
def atd(arr):
    x = numpy.square(numpy.abs(arr - man(arr)))
    return numpy.sqrt(numpy.nanmean(x))


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

  logit_three = generate_true_logistic(logit_size)    #37 and 37*3
  product1 = numpy.zeros(data.size,dtype=numpy.float64)

  for i in range(pad_before,data.size-pad_after):
    temp = data[i-pad_before:i+pad_after]
    temp = numpy.sort(temp)
    temp = (temp - temp[0])/temp[-1] ##normalize to 0,1
    product1[i] = 1 - numpy.corrcoef(temp,logit_three)[0,1]

  product1[:pad_before] = product1[pad_before:pad_before*2]
  product1[-pad_after:] = numpy.flip(product1[-pad_after*2:-pad_after])

  result = smoothpadded(product1,smooth_size)
  return result


measurements = data[0:rate]

#logit_size: int,smooth_size:int - logit size should be an odd number close to 1/4th of rate. smooth_size should be an odd number close to 4x rate.
#rate should be the sampling rate / some quantity. For 48000 I chose 100. 
#this lends? time resolution? 10ms frequency resolution? 800? idk
#time to process is close to 0.61x realtimes using numba
R = find_noise_factor(measurements,121,1921)

R[R<atd(R)] = 0 #this step is unclear
R[R>0] = 1 #after finding our noise floor, threshold all remaining values to 1
#alternatively, normalize R
result = measurements * R
#this gives *some* noise reduction on simon and dave, but does so without any kind of STFT processing



