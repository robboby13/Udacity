# The lane detection algorithm was developed on the NVIDIA TX2 embedded platform
Libraries OpenCV3.2 with Python3 Bindings

Reference the tutorial for help with installing these libraries
https://www.youtube.com/watch?v=GlDjscSAtDY

Full Git
https://github.com/jetsonhacks/buildOpenCVTX2

To run the the build file

$ ./buildOpenCV.sh

The build system has been known at times to have issues. It's worth doing a sanity check after the build is complete:

$ cd $HOME/opencv/build

$ make

This should ensure that everything has been built.

After this, you can install the new build:

$ cd $HOME/opencv/build

$ sudo make install

# Jetson TX2 (Turned Python2 OFF, Python3 ON)
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DBUILD_PNG=OFF \
    -DBUILD_TIFF=OFF \
    -DBUILD_TBB=OFF \
    -DBUILD_JPEG=OFF \
    -DBUILD_JASPER=OFF \
    -DBUILD_ZLIB=OFF \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_opencv_java=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=ON \
    -DENABLE_PRECOMPILED_HEADERS=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_OPENMP=OFF \
    -DWITH_FFMPEG=ON \
    -DWITH_GSTREAMER=OFF \
    -DWITH_GSTREAMER_0_10=OFF \
    -DWITH_CUDA=ON \
    -DWITH_GTK=ON \
    -DWITH_VTK=OFF \
    -DWITH_TBB=ON \
    -DWITH_1394=OFF \
    -DWITH_OPENEXR=OFF \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 \
    -DCUDA_ARCH_BIN=6.2 \
    -DCUDA_ARCH_PTX="" \
    -DINSTALL_C_EXAMPLES=ON \
    -DINSTALL_TESTS=ON \
    -DOPENCV_TEST_DATA_PATH=../opencv_extra/testdata \
    ../
