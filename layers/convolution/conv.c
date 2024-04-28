#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

// check if variables RAND_MEAN and RAND_STD are defined
#ifndef RAND_MEAN
#define RAND_MEAN 0.0
#endif

#ifndef RAND_STD
#define RAND_STD 1.0
#endif

typedef struct {
  int height;
  int width;
  float *data;
} Array2D;

// Conv2D layer
typedef struct {
  int C_in;
  int C_out;
  int kernel_size;
  float *weight; // (C_out, C_in, kernel_size, kernel_size)
  float *bias;   // (C_out, 1)
} Conv2D;

void gaussian(float *dest, uint size, double mean, double std) {

  for (int i = 0; i < size; ++i) {
    // Generate a random number from a uniform distribution
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;

    // Transform the random number to a Gaussian distribution
    double z0 = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
    double z1 = sqrt(-2.0 * log(u1)) * sin(2 * M_PI * u2);

    // Adjust the mean and standard deviation
    dest[i] = mean + std * z0;
  }
  return;
}

Array2D *allocate_array2d(int height, int width) {
  Array2D *arr = (Array2D *)malloc(sizeof(Array2D));
  arr->height = height;
  arr->width = width;
  arr->data = (float *)malloc(height * width * sizeof(float));
  return arr;
}

void deallocate_array2d(Array2D *arr) {
  free(arr->data);
  free(arr);
}

// Function to perform cross-correlation (convolution)
// This method is quite slow as the size grows
// TODO: use fast fourier transform instead.
Array2D *cross_correlation(Array2D *inputs, float *weight, int dim[2],
                           int weight_partition[2]) {
  Array2D *res = allocate_array2d(dim[0], dim[1]);
  for (int i = 0; i < dim[0]; i++) {
    for (int j = 0; j < dim[1]; j++) {
      float sum = 0;
      for (int k = 0; k < weight_partition[0]; k++) {   // height of the weight
        for (int l = 0; l < weight_partition[1]; l++) { // width of the weight
          sum += inputs->data[(i + k) * inputs->width + (j + l)] *
                 weight[k * weight_partition[1] + l];
        }
      }
      res->data[i * res->width + j] = sum;
    }
  }
  return res;
}

// Function to perform the forward pass of the Conv2D layer
void conv2d_forward(Conv2D *layer, Array2D *input, float *output) {
  int B = input->height;
  int H_out = input->width - layer->kernel_size + 1;
  int W_out = input->width - layer->kernel_size + 1;
  // float *output = (float *)malloc(B * layer->C_out * H_out * W_out *
  // sizeof(float)); Array2D *output = allocate_array2d(B, layer->C_out * H_out
  // * W_out);

  for (int N_i = 0; N_i < B; N_i++) {
    for (int C_out_j = 0; C_out_j < layer->C_out; C_out_j++) {
      Array2D *cross_corr_sum = allocate_array2d(H_out, W_out);
      for (int k = 0; k < layer->C_in; k++) {
        Array2D *c = cross_correlation(
            &input[N_i * input->width * input->width +
                   k * input->width * input->width],
            &layer->weight[C_out_j * layer->C_in * layer->kernel_size *
                               layer->kernel_size +
                           k * layer->kernel_size * layer->kernel_size],
            (int[]){H_out, W_out}, (int[]){C_out_j, k});
        for (int i = 0; i < H_out; i++) {
          for (int j = 0; j < W_out; j++) {
            cross_corr_sum->data[i * W_out + j] += c->data[i * W_out + j];
          }
        }
        deallocate_array2d(c);
      }
      for (int i = 0; i < H_out; i++) {
        for (int j = 0; j < W_out; j++) {
          output[N_i * layer->C_out * H_out * W_out + C_out_j * H_out * W_out +
                 i * W_out + j] =
              layer->bias[C_out_j] + cross_corr_sum->data[i * W_out + j];
        }
      }
      deallocate_array2d(cross_corr_sum);
    }
  }
  return;
}

// Function to create a Conv2D layer
Conv2D *conv2d_init(int C_in, int C_out, int kernel_size) {
  Conv2D *layer = (Conv2D *)malloc(sizeof(Conv2D));
  layer->C_in = C_in;
  layer->C_out = C_out;
  layer->kernel_size = kernel_size;
  layer->weight =
      (float *)malloc(C_out * C_in * kernel_size * kernel_size * sizeof(float));
  layer->bias = (float *)malloc(C_out);

  /*
  * This way isn't bad either

  for (int i = 0; i < C_out; i++) {
    for (int j = 0; j < C_in * kernel_size * kernel_size; j++) {
      layer->weight[i * C_in * kernel_size * kernel_size + j] =
          (float)rand() / RAND_MAX;
    }

    layer->bias.data[i] = (float)rand() / RAND_MAX;
  }
  */

  // Initialize weights and bias with random values
  gaussian(layer->weight, C_out * C_in * kernel_size * kernel_size, RAND_MEAN,
           RAND_STD);
  gaussian(layer->bias, C_out, RAND_MEAN, RAND_STD);
  return layer;
}

// Function to destroy a Conv2D layer
void free_conv2d(Conv2D *layer) {
  free(layer->weight);
  free(layer->bias);
  free(layer);
}

int main() {
  // Create a Conv2D layer
  Conv2D *layer = conv2d_init(3, 10, 3);

  // Create a sample input array
  Array2D *input = allocate_array2d(4, 3 * 32 * 32);
  gaussian(input->data, 4 * 3 * 32 * 32, 10.0, 4.);

  // TODO: how to ree a non-heap array?
  //  float output[4 * 10 * 30 * 30];

  float *output = (float *)malloc(4 * 10 * 30 * 30 * sizeof(float));

  // Perform the forward pass
  conv2d_forward(layer, input, output);

  // Print the output shape
  printf("Output shape: (4, 10, 30, 30)\n");

  // Deallocate memory
  deallocate_array2d(input);
  free(output);
  free_conv2d(layer);

  return 0;
}
