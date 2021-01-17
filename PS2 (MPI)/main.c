#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>
#include "libs/bitmap.h"
#include <mpi.h>

// Convolutional Kernel Examples, each with dimension 3,
// gaussian kernel with dimension 5

int sobelYKernel[] = {-1, -2, -1,
                       0,  0,  0,
                       1,  2,  1};

int sobelXKernel[] = {-1, -0, 1,
                      -2,  0, 2,
                      -1,  0, 1};

int laplacian1Kernel[] = { -1,  -4,  -1,
                           -4,  20,  -4,
                           -1,  -4,  -1};

int laplacian2Kernel[] = { 0,  1,  0,
                           1, -4,  1,
                           0,  1,  0};

int laplacian3Kernel[] = { -1,  -1,  -1,
                           -1,   8,  -1,
                           -1,  -1,  -1};

int gaussianKernel[] = { 1,  4,  6,  4, 1,
                         4, 16, 24, 16, 4,
                         6, 24, 36, 24, 6,
                         4, 16, 24, 16, 4,
                         1,  4,  6,  4, 1 };

char* const kernelNames[]       = { "SobelY",     "SobelX",     "Laplacian 1",    "Laplacian 2",    "Laplacian 3",    "Gaussian"     };
int* const kernels[]            = { sobelYKernel, sobelXKernel, laplacian1Kernel, laplacian2Kernel, laplacian3Kernel, gaussianKernel };
unsigned int const kernelDims[] = { 3,            3,            3,                3,                3,                5              };
float const kernelFactors[]     = { 1.0,          1.0,          1.0,              1.0,              1.0,              1.0 / 256.0    };

int const maxKernelIndex = sizeof(kernelDims) / sizeof(unsigned int);

MPI_Datatype PIXEL;


void borderExchange(bmpImage *localImage, int kernelRad, int my_rank, int mpi_size, int root, bmpImage *southRecv, bmpImage *northRecv){
  bmpImage *southSend = newBmpImage(localImage->width, kernelRad);
  bmpImage *northSend = newBmpImage(localImage->width, kernelRad);


  //Fill southSend for all except last rank
  if (my_rank != mpi_size-1){
    for (int y = 0; y < kernelRad; y++){
      for (int x = 0; x < localImage->width; x++){
        southSend->data[y][x].r = localImage->data[localImage->height - kernelRad + y][x].r;
        southSend->data[y][x].g = localImage->data[localImage->height - kernelRad + y][x].g;
        southSend->data[y][x].b = localImage->data[localImage->height - kernelRad + y][x].b;
      }
    }
  }

  //Fill northSend for all except first rank
  if (my_rank != 0){
    for (int y = 0; y < kernelRad; y++){
      for (int x = 0; x < localImage->width; x++){
        northSend->data[y][x].r = localImage->data[y][x].r;
        northSend->data[y][x].g = localImage->data[y][x].g;
        northSend->data[y][x].b = localImage->data[y][x].b;
      }
    }
  }


  //Exchange north borders for odd ranks
  if (my_rank%2 != 0){
    MPI_Sendrecv(northSend->rawdata, localImage->width * kernelRad, PIXEL, my_rank - 1, 0, northRecv->rawdata, localImage->width * kernelRad, PIXEL, my_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  //Exchange south borders for even ranks (except last rank - no south border to send)
  }else if((my_rank%2 == 0) && (my_rank != mpi_size-1)){
    MPI_Sendrecv(southSend->rawdata, localImage->width * kernelRad, PIXEL, my_rank + 1, 0, southRecv->rawdata, localImage->width * kernelRad, PIXEL, my_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  }

  //Exchange south borders for odd ranks (except last rank - no south border to send)
  if ((my_rank%2 != 0) && (my_rank != mpi_size-1)){
    MPI_Sendrecv(southSend->rawdata, localImage->width * kernelRad, PIXEL, my_rank + 1, 0, southRecv->rawdata, localImage->width * kernelRad, PIXEL, my_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  //Exhange north borders for even ranks (except first rank - no north border to send)
  }else if(my_rank%2 == 0 && my_rank != 0){
    MPI_Sendrecv(northSend->rawdata, localImage->width * kernelRad, PIXEL, my_rank - 1, 0, northRecv->rawdata, localImage->width * kernelRad, PIXEL, my_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  }

}

// Helper function to swap bmpImageChannel pointers

void swapImage(bmpImage **one, bmpImage **two) {
  bmpImage *helper = *two;
  *two = *one;
  *one = helper;
}

// Apply convolutional kernel on image data
void applyKernel(pixel **out, pixel **in, unsigned int width, unsigned int height, int *kernel, unsigned int kernelDim, float kernelFactor, bmpImage *southRecvBorder, bmpImage *northRecvBorder, int mpi_size) {
  unsigned int const kernelCenter = (kernelDim / 2);
  for (unsigned int y = 0; y < height; y++) {
    for (unsigned int x = 0; x < width; x++) {
      unsigned int ar = 0, ag = 0, ab = 0;
      for (unsigned int ky = 0; ky < kernelDim; ky++) {
        int nky = kernelDim - 1 - ky;
        for (unsigned int kx = 0; kx < kernelDim; kx++) {
          int nkx = kernelDim - 1 - kx;

          int yy = y + (ky - kernelCenter);
          int xx = x + (kx - kernelCenter);

          if (xx >= 0 && xx < (int)width){
            if (yy >= 0 && yy < (int)height){
              ar += in[yy][xx].r * kernel[nky * kernelDim + nkx];
              ag += in[yy][xx].g * kernel[nky * kernelDim + nkx];
              ab += in[yy][xx].b * kernel[nky * kernelDim + nkx];

            }else if (yy < 0 && mpi_size > 1){
              ar += northRecvBorder->data[kernelCenter + yy][xx].r * kernel[nky * kernelDim + nkx];
              ag += northRecvBorder->data[kernelCenter + yy][xx].g * kernel[nky * kernelDim + nkx];
              ab += northRecvBorder->data[kernelCenter + yy][xx].b * kernel[nky * kernelDim + nkx];

            }else if (yy >= (int)height && mpi_size > 1){
              ar += southRecvBorder->data[yy - height][xx].r * kernel[nky * kernelDim + nkx];
              ag += southRecvBorder->data[yy - height][xx].g * kernel[nky * kernelDim + nkx];
              ab += southRecvBorder->data[yy - height][xx].b * kernel[nky * kernelDim + nkx];
            }
          }
        }
      }
      if (ar || ag || ab) {
        ar *= kernelFactor;
        ag *= kernelFactor;
        ab *= kernelFactor;
        out[y][x].r = (ar > 255) ? 255 : ar;
        out[y][x].g = (ag > 255) ? 255 : ag;
        out[y][x].b = (ab > 255) ? 255 : ab;
      } else {
        out[y][x].r = 0;
        out[y][x].g = 0;
        out[y][x].b = 0;
      }
    }
  }
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
    fprintf(out, "  -k, --kernel     <kernel>        kernel index (0<=x<=%u) (2)\n", maxKernelIndex -1);
    fprintf(out, "  -i, --iterations <iterations>    number of iterations (1)\n");

    fprintf(out, "\n");
    fprintf(out, "Example: %s before.bmp after.bmp -i 10000\n", exec);
}

int main(int argc, char **argv) {
  unsigned int iterations;
  char *output = NULL;
  char *input = NULL;
  unsigned int kernelIndex;
  int ret = 0;

  int mpi_size;
  int my_rank;

  double startTime, endTime, spentTime;
    
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int imgXSize;
  int imgYSize;
  int root = 0;

  bmpImage *image = newBmpImage(0,0);

  if (my_rank == root){
  
    /*
      Parameter parsing, don't change this!
    */

    static struct option const long_options[] =  {
        {"help",       no_argument,       0, 'h'},
        {"kernel",     required_argument, 0, 'k'},
        {"iterations", required_argument, 0, 'i'},
        {0, 0, 0, 0}
    };

    static char const * short_options = "hk:i:";
    {
      char *endptr;
      int c;
      int parse;
      int option_index = 0;
      while ((c = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
        switch (c) {
        case 'h':
          help(argv[0],0, NULL);
          goto graceful_exit;
        case 'k':
          parse = strtol(optarg, &endptr, 10);
          if (endptr == optarg || parse < 0 || parse >= maxKernelIndex) {
            help(argv[0], c, optarg);
            goto error_exit;
          }
          kernelIndex = (unsigned int) parse;
          break;
        case 'i':
          iterations = strtol(optarg, &endptr, 10);
          if (endptr == optarg) {
            help(argv[0], c, optarg);
            goto error_exit;
          }
          break;
        default:
          abort();
        }
      }
    }

    if (argc <= (optind+1)) {
      help(argv[0],' ',"Not enough arugments");
      goto error_exit;
    }

    unsigned int arglen = strlen(argv[optind]);
    input = calloc(arglen + 1, sizeof(char));
    strncpy(input, argv[optind], arglen);
    optind++;

    arglen = strlen(argv[optind]);
    output = calloc(arglen + 1, sizeof(char));
    strncpy(output, argv[optind], arglen);
    optind++;

    /*
      End of Parameter parsing!
    */
    

    /*
      Create the BMP image and load it from disk.
    */

    if (image == NULL) {
      fprintf(stderr, "Could not allocate new image!\n");
      goto error_exit;
    }

    if (loadBmpImage(image, input) != 0) {
      fprintf(stderr, "Could not load bmp image '%s'!\n", input);
      freeBmpImage(image);
      goto error_exit;
    }

    imgYSize = image -> height;
    imgXSize = image -> width;

    startTime = MPI_Wtime();
    printf("Apply kernel '%s' on image with %u x %u pixels for %u iterations\n", kernelNames[kernelIndex], image->width, image->height, iterations);
  }

  MPI_Type_contiguous(3, MPI_UNSIGNED_CHAR, &PIXEL);
  MPI_Type_commit(&PIXEL);

  //Broadcast total needed parameters
  MPI_Bcast(&imgXSize, 1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&imgYSize, 1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&iterations, 1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&kernelIndex, 1, MPI_INT, root, MPI_COMM_WORLD);

  int kernelRad = kernelDims[kernelIndex]/2;
  int localChunkHeight;

  int *counts = (int*)malloc(mpi_size*sizeof(int));
  int *pixelDispls = (int*)malloc(mpi_size*sizeof(int));

  /*
    Local processors work
  */

  int sum = 0;
  int rem = imgYSize%mpi_size;
  for (int i = 0; i < mpi_size; i++) {  
      counts[i] = (imgYSize / mpi_size)*imgXSize; 	
    
    if (rem > 0) {
      counts[i] += imgXSize; 								
      rem--;
    }	 						
    pixelDispls[i] = sum;	
    sum += counts[i];				
  }

  localChunkHeight = counts[my_rank]/imgXSize;

  //Create empty matrix for chunk
  bmpImage *localChunk = newBmpImage(imgXSize, localChunkHeight);
  bmpImage *localProcChunk= newBmpImage(imgXSize, localChunkHeight);


  //Scatter data to all ranks
  MPI_Scatterv(image->rawdata, counts, pixelDispls, PIXEL, localChunk->rawdata, counts[my_rank], PIXEL, root, MPI_COMM_WORLD);
  
  //Recieve ghost cells 
  bmpImage *southRecv = newBmpImage(localChunk->width, kernelRad);
  bmpImage *northRecv = newBmpImage(localChunk->width, kernelRad);

  // Here we do the actual computation!
  // image->data is a 2-dimensional array of pixel which is accessed row first ([y][x])
  // each pixel is a struct of 3 unsigned char for the red, blue and green colour channel
  for (unsigned int i = 0; i < iterations; i ++) {
      //Exchange borders if more than one rank
      if (mpi_size != 1){
        borderExchange(localChunk, kernelRad, my_rank, mpi_size, root, southRecv, northRecv);
      }
      applyKernel(localProcChunk->data,
                  localChunk->data,
                  imgXSize,
                  localChunkHeight,
                  kernels[kernelIndex],
                  kernelDims[kernelIndex],
                  kernelFactors[kernelIndex],
                  southRecv,
                  northRecv,
                  mpi_size
      );
      swapImage(&localProcChunk, &localChunk);
  }

  freeBmpImage(localProcChunk);

  //Gather data form all ranks
  MPI_Gatherv(localChunk->rawdata, counts[my_rank], PIXEL, image->rawdata, counts, pixelDispls, PIXEL, root, MPI_COMM_WORLD);

  freeBmpImage(localChunk);

  /*
    End Local processors work
  */


  if (my_rank == root) {
    
    endTime = MPI_Wtime();
    spentTime = endTime - startTime;
    printf("Time spent: %.3f seconds\n", spentTime);

    //Write the image back to disk
    if (saveBmpImage(image, output) != 0) {
      fprintf(stderr, "Could not save output to '%s'!\n", output);
      freeBmpImage(image);
      goto error_exit;
    };
  }
  
  MPI_Finalize();

  graceful_exit:
    ret = 0;
  error_exit:
    if (input)
      free(input);
    if (output)
      free(output);
    return ret;
  };

