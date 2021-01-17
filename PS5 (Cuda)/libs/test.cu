Skip to content
Why GitHub? 
Team
Enterprise
Explore 
Marketplace
Pricing 
Search

Sign in
Sign up
davidbp13
/
TDT4200_Assignment7
100
Code
Issues
Pull requests
Actions
Projects
Security
Insights
Join GitHub today
GitHub is home to over 50 million developers working together to host and review code, manage projects, and build software together.

TDT4200_Assignment7/main.cu
@davidbp13
davidbp13 Final CUDA code
Latest commit 7aea541 on Nov 8, 2019
 History
 1 contributor
582 lines (491 sloc)  21 KB
 
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <sys/time.h>
extern "C" {
    #include "libs/bitmap.h"
}

/* Divide the problem into blocks of BLOCKX x BLOCKY threads */
#define BLOCKY 32
#define BLOCKX 32

#define ERROR_EXIT -1

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %s %d\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Convolutional Filter Examples, each with dimension 3,
// gaussian filter with dimension 5
// If you apply another filter, remember not only to exchange
// the filter but also the filterFactor and the correct dimension.

int const sobelYFilter[] = {-1, -2, -1,
                             0,  0,  0,
                             1,  2,  1};
float const sobelYFilterFactor = (float) 1.0;

int const sobelXFilter[] = {-1, -0, -1,
                            -2,  0, -2,
                            -1,  0, -1 , 0};
float const sobelXFilterFactor = (float) 1.0;


int const laplacian1Filter[] = {  -1,  -4,  -1,
                                 -4,  20,  -4,
                                 -1,  -4,  -1};

float const laplacian1FilterFactor = (float) 1.0;

int const laplacian2Filter[] = { 0,  1,  0,
                                 1, -4,  1,
                                 0,  1,  0};
float const laplacian2FilterFactor = (float) 1.0;

int const laplacian3Filter[] = { -1,  -1,  -1,
                                  -1,   8,  -1,
                                  -1,  -1,  -1};
float const laplacian3FilterFactor = (float) 1.0;


//Bonus Filter:

int const gaussianFilter[] = { 1,  4,  6,  4, 1,
                               4, 16, 24, 16, 4,
                               6, 24, 36, 24, 6,
                               4, 16, 24, 16, 4,
                               1,  4,  6,  4, 1 };

float const gaussianFilterFactor = (float) 1.0 / 256.0;


// Apply convolutional filter on image data
void applyFilter(unsigned char **out, unsigned char **in, unsigned int width, unsigned int height, int *filter, unsigned int filterDim, float filterFactor) {
  unsigned int const filterCenter = (filterDim / 2);
  for (unsigned int y = 0; y < height; y++) {
    for (unsigned int x = 0; x < width; x++) {
      int aggregate = 0;
      for (unsigned int ky = 0; ky < filterDim; ky++) {
        int nky = filterDim - 1 - ky;
        for (unsigned int kx = 0; kx < filterDim; kx++) {
          int nkx = filterDim - 1 - kx;

          int yy = y + (ky - filterCenter);
          int xx = x + (kx - filterCenter);
          if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height)
            aggregate += in[yy][xx] * filter[nky * filterDim + nkx];
        }
      }
      aggregate *= filterFactor;
      if (aggregate > 0) {
        out[y][x] = (aggregate > 255) ? 255 : aggregate;
      } else {
        out[y][x] = 0;
      }
    }
  }
}

/************************* GPU Kernel *************************/
__global__ void device_calculate(unsigned char *out, unsigned char *in, unsigned int width, unsigned int height, int *filter, unsigned int filterDim, float filterFactor) {
  //A single pixel is assigned to one tread in each of the blocks 
  int x = blockIdx.x *  blockDim.x + threadIdx.x;
  int y = blockIdx.y *  blockDim.y + threadIdx.y;
	
  //Make sure that only threads with a valid pixel compute
  if ( (x < width) && (y  < height) ){
    unsigned int const filterCenter = (filterDim / 2);
    int aggregate = 0;
    for (unsigned int ky = 0; ky < filterDim; ky++) {
      int nky = filterDim - 1 - ky;
      for (unsigned int kx = 0; kx < filterDim; kx++) {
        int nkx = filterDim - 1 - kx;

        int yy = y + (ky - filterCenter);
        int xx = x + (kx - filterCenter);
        if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height)
          aggregate += in[xx + yy * width] * filter[nky * filterDim + nkx];
      }
    }
    aggregate *= filterFactor;
    if (aggregate > 0) {
      out[x + y * width] = (aggregate > 255) ? 255 : aggregate;
    } else {
      out[x + y * width] = 0;
    }
  }
}

__global__ void device_shareInput_calculate(unsigned char *out, unsigned char *in, unsigned int width, unsigned int height, int *filter, unsigned int filterDim, float filterFactor) {
  //A single pixel is assigned to one tread in each of the blocks 
  int x = blockIdx.x *  blockDim.x + threadIdx.x;
  int y = blockIdx.y *  blockDim.y + threadIdx.y;
  
  // Move pixels into shared memory
  extern __shared__ unsigned char in_shared[];
  in_shared[(threadIdx.x + threadIdx.y *  blockDim.x)] = in[x + y * width];

  __syncthreads();

  unsigned int const filterCenter = (filterDim / 2);
  	
  //Make sure that only threads with a valid pixel compute
  if ( (x < width) && (y  < height) ){
    // If the pixel needs pixels only from its block, use shared memory
    if ( !(threadIdx.x == 0 || threadIdx.x == (blockDim.x-1) || threadIdx.y == 0 || threadIdx.y == (blockDim.y-1)) ){  
      int aggregate = 0;
      for (unsigned int ky = 0; ky < filterDim; ky++) {
        int nky = filterDim - 1 - ky;
        for (unsigned int kx = 0; kx < filterDim; kx++) {
          int nkx = filterDim - 1 - kx;

          int yy = threadIdx.y + (ky - filterCenter);
          int xx = threadIdx.x + (kx - filterCenter);
          if (xx >= 0 && xx < blockDim.x && yy >=0 && yy < blockDim.y)
            aggregate += in_shared[(xx + yy *  blockDim.x)] * filter[nky * filterDim + nkx];
        }
      }
      aggregate *= filterFactor;
      if (aggregate > 0) {
        out[x + y * width] = (aggregate > 255) ? 255 : aggregate;
      } else {
        out[x + y * width] = 0;
      }
    }
    // If the pixel needs pixels from other blocks, use global memory 
    else{
      int aggregate = 0;
      for (unsigned int ky = 0; ky < filterDim; ky++) {
        int nky = filterDim - 1 - ky;
        for (unsigned int kx = 0; kx < filterDim; kx++) {
          int nkx = filterDim - 1 - kx;

          int yy = y + (ky - filterCenter);
          int xx = x + (kx - filterCenter);
          if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height)
            aggregate += in[xx + yy *  width] * filter[nky * filterDim + nkx];
        }
      }
      aggregate *= filterFactor;
      if (aggregate > 0) {
        out[x + y * width] = (aggregate > 255) ? 255 : aggregate;
      } else {
        out[x + y * width] = 0;
      }
      
    }    
  }
}

__global__ void device_shareFilter_calculate(unsigned char *out, unsigned char *in, unsigned int width, unsigned int height, int *filter, unsigned int filterDim, float filterFactor) { 
  //A single pixel is assigned to one tread in each of the blocks 
  int x = blockIdx.x *  blockDim.x + threadIdx.x;
  int y = blockIdx.y *  blockDim.y + threadIdx.y;

  // Move filter into shared memory
  extern __shared__ int filter_shared[];
  
  /*
  for (int i = 0; i < filterDim*filterDim; i++){
    filter_shared[i] = filter[i];
  }*/
  if (threadIdx.x + threadIdx.y *  blockDim.x < filterDim * filterDim){
    filter_shared[threadIdx.x + threadIdx.y *  blockDim.x] = filter[threadIdx.x + threadIdx.y *  blockDim.x];
  }

  __syncthreads();

  unsigned int const filterCenter = (filterDim / 2);
  	
  //Make sure that only threads with a valid pixel compute
  if ( (x < width) && (y  < height) ){

      int aggregate = 0;
      for (unsigned int ky = 0; ky < filterDim; ky++) {
        int nky = filterDim - 1 - ky;
        for (unsigned int kx = 0; kx < filterDim; kx++) {
          int nkx = filterDim - 1 - kx;

          int yy = y + (ky - filterCenter);
          int xx = x + (kx - filterCenter);
          if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height)
            aggregate += in[xx + yy *  width] * filter_shared[nky * filterDim + nkx];
        }
      }
      aggregate *= filterFactor;
      if (aggregate > 0) {
        out[x + y * width] = (aggregate > 255) ? 255 : aggregate;
      } else {
        out[x + y * width] = 0;
      }
          
  }
}

__global__ void device_shareInputandFilter_calculate(unsigned char *out, unsigned char *in, unsigned int width, unsigned int height, int *filter, unsigned int filterDim, float filterFactor) {
  //A single pixel is assigned to one tread in each of the blocks 
  int x = blockIdx.x *  blockDim.x + threadIdx.x;
  int y = blockIdx.y *  blockDim.y + threadIdx.y;
  
  extern __shared__ unsigned char shared_array[];
  
  // Move pixels into shared memory
  unsigned char* in_shared = (unsigned char*) shared_array;
  in_shared[threadIdx.x + threadIdx.y *  blockDim.x] = in[x + y * width];

  __syncthreads();

  // Move filter into shared memory
  int* filter_shared = (int*)&shared_array[blockDim.x*blockDim.y];
  
  if (threadIdx.x + threadIdx.y *  blockDim.x < filterDim * filterDim){
    filter_shared[threadIdx.x + threadIdx.y *  blockDim.x] = filter[threadIdx.x + threadIdx.y *  blockDim.x];
  }

  __syncthreads();

  unsigned int const filterCenter = (filterDim / 2);
  	
  //Make sure that only threads with a valid pixel compute
  if ( (x < width) && (y  < height) ){
    // If the pixel needs pixels only from its block, use shared memory
    if ( !(threadIdx.x == 0 || threadIdx.x == (blockDim.x-1) || threadIdx.y == 0 || threadIdx.y == (blockDim.y-1)) ){  
      int aggregate = 0;
      for (unsigned int ky = 0; ky < filterDim; ky++) {
        int nky = filterDim - 1 - ky;
        for (unsigned int kx = 0; kx < filterDim; kx++) {
          int nkx = filterDim - 1 - kx;

          int yy = threadIdx.y + (ky - filterCenter);
          int xx = threadIdx.x + (kx - filterCenter);
          if (xx >= 0 && xx < blockDim.x && yy >=0 && yy < blockDim.y)
            aggregate += in_shared[xx + yy *  blockDim.x] * filter_shared[nky * filterDim + nkx];
        }
      }
      aggregate *= filterFactor;
      if (aggregate > 0) {
        out[x + y * width] = (aggregate > 255) ? 255 : aggregate;
      } else {
        out[x + y * width] = 0;
      }
    }
    // If the pixel needs pixels from other blocks, use global memory 
    else{
      int aggregate = 0;
      for (unsigned int ky = 0; ky < filterDim; ky++) {
        int nky = filterDim - 1 - ky;
        for (unsigned int kx = 0; kx < filterDim; kx++) {
          int nkx = filterDim - 1 - kx;

          int yy = y + (ky - filterCenter);
          int xx = x + (kx - filterCenter);
          if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height)
            aggregate += in[xx + yy *  width] * filter_shared[nky * filterDim + nkx];
        }
      }
      aggregate *= filterFactor;
      if (aggregate > 0) {
        out[x + y * width] = (aggregate > 255) ? 255 : aggregate;
      } else {
        out[x + y * width] = 0;
      }
      
    }    
  }
}
/*************************************************************/

/*
 * Get system time to microsecond precision (ostensibly, the same as MPI_Wtime),
 * returns time in seconds
 */
double walltime ( void ) {
  static struct timeval t;
  gettimeofday ( &t, NULL );
  return ( t.tv_sec + 1e-6 * t.tv_usec );
}

void help(char const *exec, char const opt, char const *optarg) {
  FILE *out = stdout;
  if (opt != 0) {
    out = stderr;
    if (optarg) {
      fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
    } else {
      fprintf(out, "Invalid parameter - %c\n", opt);
    }
  }
  fprintf(out, "%s [options] <input-bmp> <output-bmp>\n", exec);
  fprintf(out, "\n");
  fprintf(out, "Options:\n");
  fprintf(out, "  -i, --iterations <iterations>    number of iterations (1)\n");

  fprintf(out, "\n");
  fprintf(out, "Example: %s in.bmp out.bmp -i 10000\n", exec);
}

int main(int argc, char **argv) {
  /*
    Parameter parsing, don't change this!
   */
  unsigned int iterations = 1;
  char *output = NULL;
  char *input = NULL;
  int ret = 0;

  static struct option const long_options[] =  {
      {"help",       no_argument,       0, 'h'},
      {"iterations", required_argument, 0, 'i'},
      {0, 0, 0, 0}
  };

  static char const * short_options = "hi:";
  {
    char *endptr;
    int c;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
      switch (c) {
      case 'h':
        help(argv[0],0, NULL);
        return 0;
      case 'i':
        iterations = strtol(optarg, &endptr, 10);
        if (endptr == optarg) {
          help(argv[0], c, optarg);
          return ERROR_EXIT;
        }
        break;
      default:
        abort();
      }
    }
  }

  if (argc <= (optind+1)) {
    help(argv[0],' ',"Not enough arugments");
    return ERROR_EXIT;
  }
  input = (char *)calloc(strlen(argv[optind]) + 1, sizeof(char));
  strncpy(input, argv[optind], strlen(argv[optind]));
  optind++;

  output = (char *)calloc(strlen(argv[optind]) + 1, sizeof(char));
  strncpy(output, argv[optind], strlen(argv[optind]));
  optind++;

  /*
    End of Parameter parsing!
   */
  
  // Timing variables
  double start;
  double hosttime=0;
  double devicetime=0;

  // CUDA device properties
  cudaDeviceProp p;
  cudaSetDevice(0);
  cudaGetDeviceProperties (&p, 0);
  printf("Device compute capability: %d.%d\n", p.major, p.minor);

  
  // Create the BMP image and load it from disk.
  bmpImage *image = newBmpImage(0,0);
  if (image == NULL) {
    fprintf(stderr, "Could not allocate new image!\n");
  }

  if (loadBmpImage(image, input) != 0) {
    fprintf(stderr, "Could not load bmp image '%s'!\n", input);
    freeBmpImage(image);
    return ERROR_EXIT;
  }


  // Create a single color channel image. It is easier to work just with one color
  bmpImageChannel *imageChannel = newBmpImageChannel(image->width, image->height);
  if (imageChannel == NULL) {
    fprintf(stderr, "Could not allocate new image channel!\n");
    freeBmpImage(image);
    return ERROR_EXIT;
  }

  // Create a single color channel image. It is easier to work just with one color (CPU reference)
  bmpImageChannel *referenceImageChannel = newBmpImageChannel(image->width, image->height);
  if (referenceImageChannel == NULL) {
    fprintf(stderr, "Could not allocate new reference image channel!\n");
    freeBmpImage(image);
    return ERROR_EXIT;
  }

  // Extract from the loaded image an average over all colors - nothing else than
  // a black and white representation
  // extractImageChannel and mapImageChannel need the images to be in the exact
  // same dimensions!
  // Other prepared extraction functions are extractRed, extractGreen, extractBlue
  if(extractImageChannel(imageChannel, image, extractAverage) != 0) {
    fprintf(stderr, "Could not extract image channel!\n");
    freeBmpImage(image);
    freeBmpImageChannel(imageChannel);
    return ERROR_EXIT;
  }

  // Extract from the loaded image an average over all colors - nothing else than
  // a black and white representation
  // extractImageChannel and mapImageChannel need the images to be in the exact
  // same dimensions!
  // Other prepared extraction functions are extractRed, extractGreen, extractBlue
  if(extractImageChannel(referenceImageChannel, image, extractAverage) != 0) {
    fprintf(stderr, "Could not extract reference image channel!\n");
    freeBmpImage(image);
    freeBmpImageChannel(referenceImageChannel);
    return ERROR_EXIT;
  }

  // CPU implementation
  bmpImageChannel *processImageChannel = newBmpImageChannel(referenceImageChannel->width, referenceImageChannel->height);
  start=walltime();
  for (unsigned int i = 0; i < iterations; i ++) {
    applyFilter(processImageChannel->data, referenceImageChannel->data, referenceImageChannel->width, referenceImageChannel->height, (int *)laplacian1Filter, 3, laplacian1FilterFactor);
    //Swap the data pointers
    unsigned char ** tmp = processImageChannel->data;
    processImageChannel->data = referenceImageChannel->data;
    referenceImageChannel->data = tmp;
    unsigned char * tmp_raw = processImageChannel->rawdata;
    processImageChannel->rawdata = referenceImageChannel->rawdata;
    referenceImageChannel->rawdata = tmp_raw;
  }
  hosttime+=walltime()-start;  
  freeBmpImageChannel(processImageChannel);

  /******************************* Set up device memory *******************************/
  // Input image
  unsigned char *imageChannelGPU;
  cudaErrorCheck( cudaMalloc((void**)&imageChannelGPU, imageChannel->width * imageChannel->height * sizeof(unsigned char)) );
  cudaErrorCheck( cudaMemcpy(imageChannelGPU, imageChannel->rawdata, imageChannel->width * imageChannel->height * sizeof(unsigned char), cudaMemcpyHostToDevice) );

  // Filter 
  int *filterGPU;
  cudaErrorCheck( cudaMalloc((void**)&filterGPU, 3 * 3 * sizeof(int)) );
  cudaErrorCheck( cudaMemcpy(filterGPU, laplacian1Filter, 3 * 3 * sizeof(int), cudaMemcpyHostToDevice) );
  
  // Output image after each iteration
  unsigned char *processImageChannelGPU;
  cudaErrorCheck( cudaMalloc((void**)&processImageChannelGPU, imageChannel->width * imageChannel->height * sizeof(unsigned char)) );
  /************************************************************************************/

  /******************************* Execute GPU Kernel *******************************/
  // GPU implementation
  dim3 gridBlock(ceil(imageChannel->width/BLOCKX), ceil(imageChannel->height/BLOCKY)); //Set the number of blocks accordingly to the image size
  dim3 threadBlock(BLOCKX, BLOCKY); //Each block will have BLOCKX * BLOCKY threads 
  start=walltime();
  for (unsigned int i = 0; i < iterations; i ++) {
    // Call the kernel (Use comments to select between the basic GPU kernel and the shared memory GPU kernel
	
    //Simple GPU kernel
    //device_calculate<<<gridBlock,threadBlock>>>(processImageChannelGPU, imageChannelGPU, imageChannel->width, imageChannel->height, filterGPU, 3, laplacian1FilterFactor);

    // GPU kernel with only input array in shared memory
    //device_shareInput_calculate<<<gridBlock,threadBlock, (BLOCKX * BLOCKY * sizeof(unsigned char))>>>(processImageChannelGPU, imageChannelGPU, imageChannel->width, imageChannel->height, filterGPU, 3, laplacian1FilterFactor);

    // GPU kernel with only filter in shared memory
    //device_shareFilter_calculate<<<gridBlock,threadBlock, (3 * 3 * sizeof(int))>>>(processImageChannelGPU, imageChannelGPU, imageChannel->width, imageChannel->height, filterGPU, 3, laplacian1FilterFactor);
   
    // GPU kernel with both input and filter in GPU memory
    device_shareInputandFilter_calculate<<<gridBlock,threadBlock, (BLOCKX * BLOCKY * sizeof(unsigned char)) + (3 * 3 * sizeof(int))>>>(processImageChannelGPU, imageChannelGPU, imageChannel->width, imageChannel->height, filterGPU, 3, laplacian1FilterFactor);
    
    // Check for errors in the kernel execution
    cudaErrorCheck( cudaPeekAtLastError() );
    cudaErrorCheck( cudaDeviceSynchronize() );
	
    // Move GPU result to be used as input for the next iteration
    cudaErrorCheck( cudaMemcpy(imageChannelGPU, processImageChannelGPU, imageChannel->width * imageChannel->height * sizeof(unsigned char), cudaMemcpyDeviceToDevice) );
    
  }
  devicetime+=walltime()-start;
  
  // Get results from the GPU
  cudaErrorCheck( cudaMemcpy(imageChannel->rawdata, processImageChannelGPU, imageChannel->width * imageChannel->height * sizeof(unsigned char), cudaMemcpyDeviceToHost) );
  /**********************************************************************************/  

  /******************************* Free device memory *******************************/
  // Input image
  cudaErrorCheck( cudaFree(imageChannelGPU) );
  
  // Filter
  cudaErrorCheck( cudaFree(filterGPU) );

  // Output image
  cudaErrorCheck( cudaFree(processImageChannelGPU) );
  /**********************************************************************************/

  // Check if result is correct
  int errors=0;
  
  for(int y=0;y<imageChannel->height;y++) {
    for(int x=0;x<imageChannel->width;x++) {
      int diff=referenceImageChannel->rawdata[x + y * imageChannel->width]-imageChannel->rawdata[x + y * imageChannel->width];
      if(diff<0) diff=-diff;
      if(diff>1) {
        if(errors<10) printf("Error on pixel %d %d: expected %d, found %d\n", x,y,
			     referenceImageChannel->rawdata[x + y * imageChannel->width],
			     imageChannel->rawdata[x + y * imageChannel->width]);
	else if(errors==10) puts("...");
	  errors++;
	}
    }
  }
  if(errors>0) printf("Found %d errors.\n",errors);
  else puts("\nDevice calculations are correct.");

  // Map our single color image back to a normal BMP image with 3 color channels
  // mapEqual puts the color value on all three channels the same way
  // other mapping functions are mapRed, mapGreen, mapBlue
  if (mapImageChannel(image, imageChannel, mapEqual) != 0) {
    fprintf(stderr, "Could not map image channel!\n");
    freeBmpImage(image);
    freeBmpImageChannel(imageChannel);
    return ERROR_EXIT;
  }
  freeBmpImageChannel(imageChannel);
  freeBmpImageChannel(referenceImageChannel);

  // Write the image back to disk
  if (saveBmpImage(image, output) != 0) {
    fprintf(stderr, "Could not save output to '%s'!\n", output);
    freeBmpImage(image);
    return ERROR_EXIT;
  };

  // Print timing results
  printf("\n");
  printf("Host time: %7.3f ms\n",hosttime*1e3);
  printf("Device time: %7.3f ms\n",devicetime*1e3);
  printf("Speedup: %7.3f \n", hosttime/ devicetime);

  ret = 0;
  if (input)
    free(input);
  if (output)
    free(output);
  return ret;
};
© 2020 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About
