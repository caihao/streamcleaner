# Intrinsic Bayesian Algorithm for Shortwave Voice Noise Reduction(IBA-SVNR) 
Since i'm not good at explanations, I'll just keep this short and simple:

realtime.py is the file you want here. 

This repository contains a really good method for reducing noise in audio, for use with speech and other similar waveforms.
It is free to use, and MIT/GPL licensed.

It can be used to denoise CW as well, although it should be tweaked for that. It should not be used for denoising data modes.
It is meant for use on shortwave radio, with bandlimited single sideband signals with bandwidth of 4000hz or less.
It assumes the speech bandwidth is 3400hz or less. It does not work on a theoretical basis strongly supported, although 
contributions from an engineer to derive such a theoretical basis for this work is strongly interesting to us as this was devised solely empirically.
More explanations in the algorithm explanation file, however: the basic gist of it is this:

It works on the basis of logit comparisons for time bins and thresholding using robust measures, with minimal dependencies.
The code itself is additionally MIT and GPL licensed, depending on numba, numpy, some library's stfts, and, for specific use cases,
invoking dearpygui, tk, pyaudio, and other components as needed, but they are not required for the core algorithm to function.
OLA-consistent, dft-centered, optimal, numba-compatible code was adapted from ssqueezepy and pyroomacoustics methods.

For the c++, everything has been ported, but has not been tested- only compiled and verified to run, and manually gone over with a fine tooth comb.
results are not satisfactorially identical to the python,  and I am too frustrated to continue different things trying to fix it.



