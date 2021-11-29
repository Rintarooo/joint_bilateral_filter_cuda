# JointBilateralFilter
JointBilateralFilter implementation on GPU(each pixel can be processed in parallel) and CPU(OpenCV ximgproc module)

## Usage

```bash
./run.sh
./build/main
```

## Result
parameters
```cpp
sigma_color = 2.0/255.0, sigma_spatial = 60.0

radius = sigma_spatial*1.5 = 90

```

outputs
```bash
joint_bilateral_filter_cpu
elapsed time(milliseconds): 2466

joint_bilateral_filter_gpu
elapsed time(milliseconds): 17502

bilateral_filter_gpu
elapsed time(milliseconds): 224
```