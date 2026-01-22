#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Structure to represent the MaxPool2d layer
typedef struct MaxPool2d {
  int kernel_size;
  int stride;
  void (*forward)(struct MaxPool2d *layer, float *inputs, int B, int C,
                  int H_in, int W_in, float *output);

  void (*backward)(struct MaxPool2d *layer, float *inputs, int B, int C,
                   int H_in, int W_in, float *output, float *grad_output,
                   float *grad_input);
} MaxPool2d;

// Forward pass
void max_pool_2d_forward(MaxPool2d *layer, float *inputs, int B, int C,
                         int H_in, int W_in, float *output) {
  int H_out = (H_in / layer->kernel_size);
  int W_out = (W_in / layer->kernel_size);

  for (int b = 0; b < B; b++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < H_out; h++) {
        for (int w = 0; w < W_out; w++) {
          int posh = h * layer->stride;
          int posw = w * layer->stride;
          float max_val = -INFINITY;
          for (int i = 0; i < layer->kernel_size; i++) {
            for (int j = 0; j < layer->kernel_size; j++) {
              float val = inputs[(b * C * H_in * W_in) + (c * H_in * W_in) +
                                 ((posh + i) * W_in) + (posw + j)];
              if (val > max_val) {
                max_val = val;
              }
            }
          }
          output[(b * C * H_out * W_out) + (c * H_out * W_out) + (h * W_out) +
                 w] = max_val;
        }
      }
    }
  }
}

// Backward pass
void max_pool_2d_backward(MaxPool2d *layer, float *inputs, int B, int C,
                          int H_in, int W_in, float *output, float *grad_output,
                          float *grad_input) {
  int H_out = (H_in / layer->kernel_size);
  int W_out = (W_in / layer->kernel_size);

  for (int b = 0; b < B; b++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < H_out; h++) {
        for (int w = 0; w < W_out; w++) {
          int posh = h * layer->stride;
          int posw = w * layer->stride;
          float max_val = -INFINITY;
          int max_idx_i = -1;
          int max_idx_j = -1;
          for (int i = 0; i < layer->kernel_size; i++) {
            for (int j = 0; j < layer->kernel_size; j++) {
              float val = inputs[(b * C * H_in * W_in) + (c * H_in * W_in) +
                                 ((posh + i) * W_in) + (posw + j)];
              if (val > max_val) {
                max_val = val;
                max_idx_i = i;
                max_idx_j = j;
              }
            }
          }
          grad_input[(b * C * H_in * W_in) + (c * H_in * W_in) +
                     ((posh + max_idx_i) * W_in) + (posw + max_idx_j)] +=
              grad_output[(b * C * H_out * W_out) + (c * H_out * W_out) +
                          (h * W_out) + w];
        }
      }
    }
  }
}

MaxPool2d maxpool_2d_init(int kernel_size, int stride) {
  MaxPool2d layer;
  layer.kernel_size = kernel_size;
  layer.stride = stride;
  layer.forward = max_pool_2d_forward;
  layer.backward = max_pool_2d_backward;
  return layer;
}

int main() {
  // Example usage
  int B = 1;
  int C = 3;
  int H_in = 4;
  int W_in = 4;
  int kernel_size = 2;
  int stride = 2;

  float *inputs = (float *)malloc(B * C * H_in * W_in * sizeof(float));
  float *output = (float *)malloc(B * C * (H_in / kernel_size) *
                                  (W_in / kernel_size) * sizeof(float));
  float *grad_output = (float *)malloc(B * C * (H_in / kernel_size) *
                                       (W_in / kernel_size) * sizeof(float));
  float *grad_input = (float *)malloc(B * C * H_in * W_in * sizeof(float));

  // Initialize inputs and grad_output
  for (int i = 0; i < B * C * H_in * W_in; i++) {
    inputs[i] = (float)i;
  }
  for (int i = 0; i < B * C * (H_in / kernel_size) * (W_in / kernel_size);
       i++) {
    grad_output[i] = (float)i;
  }

  MaxPool2d layer = maxpool_2d_init(kernel_size, stride);

  layer.forward(&layer, inputs, B, C, H_in, W_in, output);
  layer.backward(&layer, inputs, B, C, H_in, W_in, output, grad_output,
                 grad_input);

  // Print output and grad_input
  for (int i = 0; i < B * C * (H_in / kernel_size) * (W_in / kernel_size);
       i++) {
    printf("%f ", output[i]);
  }
  printf("\n");
  for (int i = 0; i < B * C * H_in * W_in; i++) {
    printf("%f ", grad_input[i]);
  }
  printf("\n");

  free(inputs);
  free(output);
  free(grad_output);
  free(grad_input);

  return 0;
}
