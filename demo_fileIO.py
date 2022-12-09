'''
Demo v1.1
Copyright 2022 Oscar Steila, Joshuah Rainstar
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
'''
Copyright 2022 Oscar Steila, Joshuah Rainstar

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

#How to use this file:
#step one: using https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Windows-x86_64.exe
#install python.
#step three: locate the dedicated python terminal in your start menu, called mambaforge prompt.
#within that prompt, give the following instructions:
#conda install pip numpy scipy
#pip install librosa tk dearpygui np-rw-buffer numba
#if all of these steps successfully complete, you're ready to go, otherwise fix things.
#run it with python demo_fileIO.py



import os
import numpy

from time import sleep
from time import time as time
from librosa import stft, istft
from scipy.io import wavfile
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def hann(M, sym=True):
    a = [0.5, 0.5]
    fac = numpy.linspace(-numpy.pi, numpy.pi, M)
    w = numpy.zeros(M)
    for k in range(len(a)):
        w += a[k] * numpy.cos(k * fac)
    return w

def boxcar(M, sym=True):
    a = [0.5, 0.5]
    fac = numpy.linspace(-numpy.pi, numpy.pi, M)
    w = numpy.zeros(M)
    for k in range(len(a)):
        w += a[k] * numpy.cos(k * fac)
        return w

def moving_average(x, w):
    return numpy.convolve(x, numpy.ones(w), 'same') / w


def man(arr):

    med = numpy.nanmedian(arr[numpy.nonzero(arr)])
    return numpy.nanmedian(numpy.abs(arr - med))

def atd(arr):
    x = numpy.square(abs(arr - man(arr)))
    return numpy.sqrt(numpy.nanmean(x))

def threshhold(arr):
  return (atd(arr)+ numpy.nanmedian(arr[numpy.nonzero(arr)])) 

def corrected_logit(size):
    fprint = numpy.linspace(0, 1, size)
    fprint [1:-1] /= 1 - fprint [1:-1]
    fprint [1:-1] = numpy.log(fprint [1:-1])
    fprint[0] = -6
    fprint[-1] = 6
    return fprint

#precalculate the logistic function for our entropy calculations.
#save some cycles with redundancy.
#change this value if you change your entropy window.
logit = corrected_logit(32)

def entropy(data: numpy.ndarray):
    a = numpy.sort(data)
    scaled = numpy.interp(a, (a[0], a[-1]), (-6, +6))
    z = numpy.corrcoef(scaled, logit)
    completeness = z[0, 1]
    sigma = 1 - completeness
    return sigma



    
def runningMeanFast(x, N):
    return numpy.convolve(x, numpy.ones((N,))/N,mode="valid")

#depending on presence of openblas, as fast as numba.  
def numpy_convolve_filter(data: numpy.ndarray):
   normal = data.copy()
   transposed = data.copy()
   transposed = transposed.T
   transposed_raveled = numpy.ravel(transposed)
   normal_raveled = numpy.ravel(normal)

   A =  runningMeanFast(transposed_raveled, 3)
   transposed_raveled[0] = (transposed_raveled[0] + (transposed_raveled[1] + transposed_raveled[2]) / 2) /3
   transposed_raveled[-1] = (transposed_raveled[-1] + (transposed_raveled[-2] + transposed_raveled[-3]) / 2)/3
   transposed_raveled[1:-1] = A 
   transposed = transposed.T


   A =  runningMeanFast(normal_raveled, 3)
   normal_raveled[0] = (normal_raveled[0] + (normal_raveled[1] + normal_raveled[2]) / 2) /3
   normal_raveled[-1] = (normal_raveled[-1] + (normal_raveled[-2] + normal_raveled[-3]) / 2)/3
   normal_raveled[1:-1] = A
   return (transposed + normal )/2


def numpyfilter_wrapper_50(data: numpy.ndarray):
  d = data.copy()
  for i in range(50):
    d = numpy_convolve_filter(d)
  return d


def denoise_old(data: numpy.ndarray):
    data= numpy.asarray(data,dtype=float) #correct byte order of array   

    stft_r = stft(data,n_fft=512,window=boxcar) #get complex representation
    stft_vr = numpy.square(stft_r.real) + numpy.square(stft_r.imag) #obtain absolute**2
    Y = stft_vr[~numpy.isnan(stft_vr)]
    max = numpy.where(numpy.isinf(Y),0,Y).argmax()
    max = Y[max]
    stft_vr = numpy.nan_to_num(stft_vr, copy=True, nan=0.0, posinf=max, neginf=0.0)#correct irrationalities
    stft_vr = numpy.sqrt(stft_vr) #stft_vr >= 0 
    stft_vr=(stft_vr-numpy.nanmin(stft_vr))/numpy.ptp(stft_vr)
    ent1 = numpy.apply_along_axis(func1d=entropy,axis=0,arr=stft_vr[0:32,:]) #32 is pretty much the speech cutoff?
    ent1 = ent1 - numpy.min(ent1)

    t = threshhold(stft_vr)     
    mask_one = numpy.where(stft_vr>=t, 1,0)
    stft_demo = numpy.where(mask_one == 0, stft_vr,0)
    stft_d = stft_demo.flatten()
    stft_d = stft_d[stft_d>0]
    r = man(stft_d) #obtain a noise background basis
    
    stft_r = stft(data,n_fft=512,window=hann) #get complex representation
    
    stft_vr = numpy.square(stft_r.real) + numpy.square(stft_r.imag) #obtain absolute**2
    Y = stft_vr[~numpy.isnan(stft_vr)]
    max = numpy.where(numpy.isinf(Y),-numpy.Inf,Y).argmax()
    max = Y[max]
    stft_vr = numpy.nan_to_num(stft_vr, copy=True, nan=0.0, posinf=max, neginf=0.0)#correct irrationalities
    stft_vr = numpy.sqrt(stft_vr) #stft_vr >= 0 
    stft_vr=(stft_vr-numpy.nanmin(stft_vr))/numpy.ptp(stft_vr) #normalize to 0,1
   
    ent = numpy.apply_along_axis(func1d=entropy,axis=0,arr=stft_vr[0:32,:]) #32 is pretty much the speech cutoff?
    ent = ent - numpy.min(ent)
    ent  = moving_average(ent,14)
    ent1  = moving_average(ent1,14)
    #seems to be a reasonable compromise
    minent = numpy.minimum(ent,ent1)
    minent=(minent-numpy.nanmin(minent))/numpy.ptp(minent)#correct basis
    maxent = numpy.maximum(ent,ent1)
    
    trend = moving_average(maxent,20)
    factor = numpy.max(trend)
    if factor < 0.0577215664: #unknown the exact most precise, correct option. 
    #this step does a "false alert" for a frame which is either containing some signal or no signal.
    #generally, anything which is above 0.067 is pretty much guaranteed to contain signal. Anything below 0.55 is guaranteed to be noise.
    #based on my calculations, 0.56 passes 10% of noise, 0.057 passes 1%. But these settings also pass 100% of signal.
    #0.058 begins to miss a few of the hardest to read voice segments, normally illegible.

      stft_r = stft_r * r
      processed = istft(stft_r,window=hann)
      return processed
      #no point wasting cycles smoothing information which isn't there!

    maxent=(maxent-numpy.nanmin(maxent))/numpy.ptp(maxent)#correct basis 

    ent = (maxent+minent)
    ent = ent - numpy.min(ent)
    trend=(ent-numpy.nanmin(ent))/numpy.ptp(ent)#correct basis 

    t1 = atd(trend)/2 #unclear where to set this. too aggressive and it misses parts of syllables.
    trend[trend<t1] = 0
    trend[trend>0] = 1
    t = (threshhold(stft_vr[stft_vr>=t]) - atd(stft_vr[stft_vr>=t]) ) +man(stft_vr)   #obtain the halfway threshold
    #note: this threshhold is still not perfectly refined, but had to be optimized for a variety of SNR.
    mask_two = numpy.where(stft_vr>=t, 1.0,0)

    mask = mask_two * trend[None,:] #remove regions from the mask that are noise
    r = r * factor
    mask[mask==0] = r #reduce warbling, you could also try r/2 or r/10 or something like that, its not as important
    mask = numpyfilter_wrapper_50(mask)
    
    mask=(mask-numpy.nanmin(mask))/numpy.ptp(mask)#correct basis    

    stft_r = stft_r * mask
    processed = istft(stft_r,window=hann)
    return processed


import numba
@numba.jit()
def entropy_numba(data: numpy.ndarray):
    a = numpy.sort(data)
    scaled = numpy.interp(a, (a[0], a[-1]), (-6, +6))
    z = numpy.corrcoef(scaled, logit)
    completeness = z[0, 1]
    sigma = 1 - completeness
    return sigma


def numpyentropycheck(data: numpy.ndarray):
  d = data.copy()
  raveled = numpy.ravel(d)
  windows =  numpy.lib.stride_tricks.sliding_window_view(raveled, window_shape = 32)
  raveled[0:windows.shape[0]] = numpy.apply_along_axis(func1d = entropy_numba,axis=1,arr=windows)
  return d  

#here is an experimental new denoising algorithm. 
#it may be less sensitive than the previous version, but it is computationally competitive,
#and does more. It may be more robust.
def denoise(data: numpy.ndarray):
    data= numpy.asarray(data,dtype=float) #correct byte order of array   

    stft_r = stft(data,n_fft=512,window=boxcar) #get complex representation
    stft_vr =  numpy.abs(stft_r) #returns the same as other methods
    stft_vr=(stft_vr-numpy.nanmin(stft_vr))/numpy.ptp(stft_vr) #normalize to 0,1
    window = stft_vr[0:32,:]
    window = numpy.pad(window,((0,0),(32,32)),mode="symmetric")
    e = numpyentropycheck(window.T).T[:,32:-32]
    entropy = numpy.apply_along_axis(func1d=numpy.max,axis=0,arr=e)
    o = numpy.pad(entropy, entropy.size//2, mode='median')
    entropy = moving_average(o,14)[entropy.size//2: -entropy.size//2]
    factor = numpy.sum(entropy)/entropy.size

    stft_r = stft(data,n_fft=512,window=hann) #get complex representation
    stft_vr =  numpy.abs(stft_r) #returns the same as other methods
    stft_vr=(stft_vr-numpy.nanmin(stft_vr))/numpy.ptp(stft_vr) #normalize to 0,1
    residue = man(stft_vr)  
    if factor < 0.0747597920253411435178730:  #Renyi's parking constant m 
    #this number may be robust enough to trust.
      stft_r = stft_r * residue #return early, and terminate the noise
      processed = istft(stft_r,window=hann)
      return processed

    floor = threshhold(stft_vr)  

    entropy_threshhold = 0.0834626841674073186814297 #AGM
    #this number may need to be reduced.

    entropy[entropy<entropy_threshhold] = 0
    entropy[entropy>0] = 1

    threshold = (threshhold(stft_vr[stft_vr>=floor]) - atd(stft_vr[stft_vr>=floor]))
    mask_two = numpy.where(stft_vr>=threshold, 1.0,0)


    mask = mask_two * entropy[None,:] #remove regions from the mask that are noise
    residue = residue * factor
    mask[mask==0] = residue 
    mask = numpyfilter_wrapper_50(mask)
    
    mask=(mask-numpy.nanmin(mask))/numpy.ptp(mask)#correct basis    

    stft_r = stft_r * mask
    processed = istft(stft_r,window=hann)
    return processed



def padarray(A, size):
    t = size - len(A)
    return numpy.pad(A, pad_width=(0, t), mode='constant',constant_values=numpy.std(A))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def process_data(data: numpy.ndarray):
    print("processing ", data.size / rate, " seconds long file at ", rate, " rate.")
    start = time()
    processed = []
    for each in chunks(data, rate):
        if each.size == rate:
            processed.append(denoise(each))
        else:
            psize = each.size
            working = padarray(each, rate)
            working = denoise(working)
            processed.append(working[0:psize])
    end = time()
    print("took ", end - start, " to process ", data.size / rate)
    return numpy.concatenate((processed), axis=0)   

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
# look for input file.wav
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename(filetypes=(('wav files', '*.wav'),), title='Select input file.wav')
    file_path, file_tail = os.path.split(filename)
    infile = filename
    outfile = f"{file_path}/cleanup, {file_tail}"
    print(infile)
    print(outfile)
    rate, data = wavfile.read(infile) # pcm16bit
    data_max = (2 ** 15 - 1)  # peak  in pcm16bit
  # peak level in pcm16bit
    data = data * 1.0 / data_max  # from pcm16bit to float (-1.0, 1.0)

    reduced_noise = process_data(data) *2.0  #  6 db gain
    # from float (-1.0, 1.0) to pcm16bit
    numpy.clip(reduced_noise, -1.0, 1.0)   # clip signal float range (1.0,-1-0)
    reduced_noise = reduced_noise * data_max
    wavfile.write(outfile, rate, reduced_noise.astype(numpy.int16))

