# HCGrid-ROCm

- Version: 1.0.0
- Authors: 

## Introduction

HCGrid is a high performance gridding framework for the large single-dish radio telescope. This is the ROCm version of HCGrid. Please visit https://github.com/HWang-Summit/HCGrid for more details.

## Dependencies

We kept the dependencies as minimal as possible. The following packages are required:
- cfitsio-3.47 or later
- wcslib-5.16 or later
- ROCm toolkit

 All of these packages can be found in "Dependencies" directory or get from follow address:

- cfitsio: https://heasarc.gsfc.nasa.gov/fitsio/
- wcslib: https://www.atnf.csiro.au/people/Mark.Calabretta/WCS/
- ROCm: https://github.com/RadeonOpenCompute/ROCm

**Note:** *Not all AMD GPUs can deploy ROCm, many early architectures are not supported by ROCm, please refer to the ROCm documentation for details (https://github.com/RadeonOpenCompute/ROCm).*

## Compile

