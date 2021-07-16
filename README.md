# HCGrid-ROCm

New emerging large single-dish radio telescopes like Five-hundred-meter Aperture Spherical radio Telescope (FAST) have heightened the need for developing both more resilient hardware and efficient algorithms than those conventional technology can provide. To process the spectral line data from the radio telescopes, the convolution-based gridding algorithm is widely adopted to solve the most compute-intensive parts of creating sky images: gridding.

HCGrid is a high performance gridding software for the spectral line data gridding of the large single-dish radio telescope. There are two versions of HCGrid, based on ROCm and CUDA, which can be deployed on heterogeneous computing environments with AMD GPUs and NVIDIA GPUs. The current version is ***HCGrid-ROCm***.
## More About HCGrid-ROCm

More details please visit the **HCGrid-CUDA** repository (https://github.com/HWang-Summit/HCGrid), here are some points to be noted when using HCGrid-ROCm.

### Hardware and Software Support

ROCm is focused on using AMD GPUs to accelerate computational tasks, not all AMD GPUs can deploy ROCm, many early architectures are not supported by ROCm. To get more details about hardware and software support of ROCm, please refer to the ROCm documentation for details (https://github.com/RadeonOpenCompute/ROCm).



