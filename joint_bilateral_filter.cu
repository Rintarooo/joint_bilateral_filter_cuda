#include "joint_bilateral_filter.cuh"

static texture<float4, cudaTextureType2D, cudaReadModeElementType> jointTex;
static texture<float, cudaTextureType2D, cudaReadModeElementType> srcTex;

static __global__ void applyJointBilateral(float* dst, float sigma_color, float sigma_spatial, int radius, int rows, int cols)
{
  const int u_ = blockIdx.x * blockDim.x + threadIdx.x;
  const int v_ = blockIdx.y * blockDim.y + threadIdx.y;
  const int i  = u_ + v_*cols;
  
  if(u_ < radius || cols-radius <= u_ || v_ < radius || rows-radius <= v_){
    dst[i] = 0.0;
  }
  else{
    // const float4 p = tex2D(jointTex, u_+0.5, v_+0.5);
    const float p_dash = tex2D(srcTex, u_+0.5, v_+0.5);
              
    float sum = 0.0;
    float sumw = 0.0;
    const float den_spatial = 2*sigma_spatial*sigma_spatial;
    const float den_color = 2*sigma_color*sigma_color; 
    for(int uu = -radius; uu <= radius; uu++){
        for(int vv = -radius; vv <= radius; vv++){
              const float w_spatial = __expf(-(uu*uu + vv*vv) / den_spatial);
              // const float4 q = tex2D(jointTex, u_+uu+0.5, v_+vv+0.5);
              // const float id = (fabsf(p.x-q.x) + fabsf(p.y-q.y) + fabsf(p.z-q.z))/3.0;
              // const float id = fabsf(p.x-q.x);
              const float src_depth = tex2D(srcTex, u_+uu+0.5, v_+vv+0.5);
              const float id = fabsf(p_dash-src_depth);
              const float w_color = __expf(-(id*id) / den_color);
              const float w = w_spatial * w_color;
              // const float src_depth = tex2D(srcTex, u_+uu+0.5, v_+vv+0.5);
              // const float src_depth = src[i];
              sum += w * src_depth;
              sumw += w;
        }
     }
    dst[i] = sum / sumw;
    // const float src_depth = tex2D(srcTex, u_+0.5, v_+0.5);
    // dst[i] = src_depth;
  }
  // const float4 p = tex2D(jointTex, u_+0.5, v_+0.5);
  // dst[i] = (p.x+p.y+p.z)/3.0;    
}


void applyJointBilateralCaller(float*src, float* dst, float4* joint, int rows, int cols, int imageStep_src, int imageStep_joint,
  float sigma_color, float sigma_spatial, int radius)
{
  // TODO set dimBlock based on warp size
  dim3 dimBlock(16, 16);
  dim3 dimGrid((cols  + dimBlock.x - 1) / dimBlock.x,
               (rows + dimBlock.y - 1) / dimBlock.y);
 
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat); 
  jointTex.normalized     = false;
  jointTex.addressMode[0] = cudaAddressModeClamp;  // out of border references return first or last element
  jointTex.addressMode[1] = cudaAddressModeClamp;
  jointTex.filterMode     = cudaFilterModeLinear;
  CV_CUDEV_SAFE_CALL(cudaBindTexture2D(0, jointTex, joint, channelDesc, cols, rows, imageStep_joint));
  
  cudaChannelFormatDesc channelDesc_float1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); 
  srcTex.normalized     = false;
  srcTex.addressMode[0] = cudaAddressModeClamp;  // out of border references return first or last element
  srcTex.addressMode[1] = cudaAddressModeClamp;
  srcTex.filterMode     = cudaFilterModeLinear;
  CV_CUDEV_SAFE_CALL(cudaBindTexture2D(0, srcTex, src, channelDesc_float1, cols, rows, imageStep_src));

  CV_CUDEV_SAFE_CALL(cudaGetLastError());
  CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());

  applyJointBilateral<<<dimGrid, dimBlock>>>(dst, sigma_color, sigma_spatial, radius, rows, cols);

  CV_CUDEV_SAFE_CALL(cudaGetLastError());
  CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
  CV_CUDEV_SAFE_CALL(cudaUnbindTexture(jointTex));
  CV_CUDEV_SAFE_CALL(cudaUnbindTexture(srcTex));

}