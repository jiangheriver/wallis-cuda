# wallis-cuda

Accelerate wallis filter process via CUDA

Please visit my homepage: https://jiangheriver.github.io

1.In this stage, raster image for input is restricted to single channel and 16 bits per pixel, output image would be 8 bits per pixel.

2.Maximum width of input image is limited to 40000. (defined via REQUIRE_SUPPORT_RASTER_WIDTH)

3.The filteri's neighbor window size is restricted to 31*31, due to the maximum parrell ability for CUDA thread block (Maximum number of threads per block = 1024, when CUDA compute capability >= 2.0)

4.GPU computed capablity should >= 2.0 (Fermi microarchitecture or newer), due to above item and Maximum amount of shared memory per multiprocessor should >= 48KB

5.Please compile the source code with VS2013 / GCC 4.8.x or higher, due to c++11 support

