# Cloth Animation with Time-dependent Persistent Wrinkles
In our Eurographics 2025 paper <em>Cloth Animation with Time-dependent Persistent Wrinkles</em>, we propose a noval method for simulating the cloth wrinkles' time-dependence obversed on the real clothes.

![image](/images/teaser.jpg)

By modeling both cloth internal friction and plasticity, our simulator can simulate the wrinkles resulted from different deformations. We'd like to open source our code and data here for future studies.

## Compiling

The program was developed by using x64 MSVC (19.36.32537) and C++ 20 standard. It also needs the extra libraries listed below:

- CUDA
- Eigen
- Alglib
- JSON-cpp
