#include<stdio.h>
__global__ void square(float *d_out,float *d_in){
    int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f*f;
}
int main(){
    const int size = 64;
    const int bytes = size*sizeof(float);
    float h_in[size];
    for (int i = 0; i<size; i++){
        h_in[i] = float(i);
    }
    float h_out[size];
    float *d_in;
    float *d_out;
    cudaMalloc((void **)&d_in,bytes);
    cudaMalloc((void **)&d_out,bytes);
    cudaMemcpy(d_in,h_in,bytes,cudaMemcpyHostToDevice);
    square<<<1 , size>>>(d_out,d_in);
    cudaMemcpy(h_out,d_out,bytes,cudaMemcpyDeviceToHost);
    for(int i = 0; i<size;i++){
        printf("%f",h_out[i]);
        printf(((i%4) != 3)?"\t":"\n");
    }
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}