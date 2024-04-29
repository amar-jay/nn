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
  int batch_size;
  int height;
  int width;
  float *grad;
  float *data;
} Array2D;

typedef struct {
  int batch_size;
  int seq_len;
  float *data;
  float *grad;
} Array1D;

// Conv1D layer
typedef struct Conv1D {
  int in_channels;
  int out_channels;
  int kernel_size;
  int stride;
  float *weights;
  float *bias;
  float *grad_weights;
  float *grad_bias;

  void (*forward)(struct Conv1D *layer, Array1D inputs,
                  Array1D output); // (inputs, output)

  void (*backward)(struct Conv1D *layer, Array1D inputs, float *grad_output,
                   float *grad_input); //(inputs, grad_output, grad_input)
} Conv1D;

typedef struct Conv2D {
  int in_channels;
  int out_channels;
  int kernel_size;
  int stride;
  float *weights;
  float *bias;
  float *grad_weights;
  float *grad_bias;

  void (*forward)(struct Conv2D *layer, Array2D inputs, Array2D output);

  void (*backward)(struct Conv2D *layer, Array2D inputs, float *grad_output,
                   float *grad_input);
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

Array1D *allocate_array1d(int batch_size, int channel, int seq_len) {
  Array1D *arr = (Array1D *)malloc(sizeof(Array1D));
  arr->batch_size = batch_size;
  arr->seq_len = seq_len;
  arr->data = (float *)calloc(batch_size * channel * seq_len, sizeof(float));
  arr->grad = (float *)calloc(batch_size * channel * seq_len, sizeof(float));
  return arr;
}

void deallocate_array1d(Array1D *arr) {
  free(arr->data);
  free(arr);
}

Array2D *allocate_array2d(int batch_size, int channel, int height, int width) {
  Array2D *arr = (Array2D *)malloc(sizeof(Array2D));
  arr->batch_size = batch_size;
  arr->height = height;
  arr->width = width;
  arr->data =
      (float *)calloc(batch_size * channel * height * width, sizeof(float));

  arr->grad =
      (float *)calloc(batch_size * channel * height * width, sizeof(float));
  return arr;
}

void deallocate_array2d(Array2D *arr) {
  free(arr->data);
  free(arr);
}

// Function to perform cross-correlation (convolution)
// This method is quite slow as the size grows
// TODO: use fast fourier transform instead.
void cross_correlation(float *inputs, float *weight, int dim[2],
                       int inputs_partition[2], int weight_partition[2],
                       float *out) {
  float *res = (float *)malloc(dim[0] * dim[1] * sizeof(float));
  for (int i = 0; i < dim[0]; i++) {
    for (int j = 0; j < dim[1]; j++) {
      float sum = 0;
      for (int k = 0; k < weight_partition[0]; k++) {   // height of the weight
        for (int l = 0; l < weight_partition[1]; l++) { // width of the weight
          sum += inputs[(i + k) * inputs_partition[1] + (j + l)] *
                 weight[k * weight_partition[1] + l];
        }
      }
      res[i * dim[1] + j] = sum;
    }
  }
}

void conv1d_forward(Conv1D *layer, Array1D inputs, Array1D output) {
  int B = inputs.batch_size;
  int out_seq_len = (inputs.seq_len - layer->kernel_size) / layer->stride + 1;
  for (int b = 0; b < B; b++) {
    for (int o = 0; o < layer->out_channels; o++) {
      for (int i = 0; i < out_seq_len; i++) {
        float sum = 0.0;
        for (int k = 0; k < layer->kernel_size; k++) {
          for (int c = 0; c < layer->in_channels; c++) {
            sum += inputs.data[b * layer->in_channels * inputs.seq_len +
                               c * inputs.seq_len + i * layer->stride + k] *
                   layer->weights[o * layer->in_channels * layer->kernel_size +
                                  c * layer->kernel_size + k];
          }
        }
        sum += layer->bias[o];
        output
            .data[b * layer->out_channels * out_seq_len + o * out_seq_len + i] =
            sum;
      }
    }
  }
}

// Backward pass
void conv1d_backward(Conv1D *layer, Array1D inputs, float *grad_output,
                     float *grad_input) {
  int out_seq_len = (inputs.seq_len - layer->kernel_size) / layer->stride + 1;

  // Compute grad_input
  for (int b = 0; b < inputs.batch_size; b++) {
    for (int c = 0; c < layer->in_channels; c++) {
      for (int i = 0; i < inputs.seq_len; i++) {
        float sum = 0.0;
        for (int o = 0; o < layer->out_channels; o++) {
          for (int k = 0; k < layer->kernel_size; k++) {
            if (i - k * layer->stride >= 0 &&
                i - k * layer->stride < out_seq_len) {
              sum +=
                  grad_output[b * layer->out_channels * out_seq_len +
                              o * out_seq_len + i - k * layer->stride] *
                  layer->weights[o * layer->in_channels * layer->kernel_size +
                                 c * layer->kernel_size + k];
            }
          }
        }
        grad_input[b * layer->in_channels * inputs.seq_len +
                   c * inputs.seq_len + i] = sum;
      }
    }
  }

  // Compute grad_weights
  for (int o = 0; o < layer->out_channels; o++) {
    for (int c = 0; c < layer->in_channels; c++) {
      for (int k = 0; k < layer->kernel_size; k++) {
        float sum = 0.0;
        for (int b = 0; b < inputs.batch_size; b++) {
          for (int i = 0; i < out_seq_len; i++) {
            sum += grad_output[b * layer->out_channels * out_seq_len +
                               o * out_seq_len + i] *
                   inputs.data[b * layer->in_channels * inputs.seq_len +
                               c * inputs.seq_len + i * layer->stride + k];
          }
        }
        layer->grad_weights[o * layer->in_channels * layer->kernel_size +
                            c * layer->kernel_size + k] = sum;
      }
    }
  }

  // Compute grad_bias
  for (int o = 0; o < layer->out_channels; o++) {
    float sum = 0.0;
    for (int b = 0; b < inputs.batch_size; b++) {
      for (int i = 0; i < out_seq_len; i++) {
        sum += grad_output[b * layer->out_channels * out_seq_len +
                           o * out_seq_len + i];
      }
    }
    layer->grad_bias[o] = sum;
  }
}

void conv2d_forward(Conv2D *layer, Array2D inputs, Array2D output) {
  int out_height = (inputs.height - layer->kernel_size) / layer->stride + 1;
  int out_width = (inputs.width - layer->kernel_size) / layer->stride + 1;
  for (int b = 0; b < inputs.batch_size; b++) {
    for (int o = 0; o < layer->out_channels; o++) {
      for (int i = 0; i < out_height; i++) {
        for (int j = 0; j < out_width; j++) {
          float sum = 0.0;
          for (int k = 0; k < layer->kernel_size; k++) {
            for (int l = 0; l < layer->kernel_size; l++) {
              for (int c = 0; c < layer->in_channels; c++) {
                sum +=
                    inputs.data[b * layer->in_channels * inputs.height *
                                    inputs.width +
                                c * inputs.height * inputs.width +
                                (i * layer->stride + k) * inputs.width +
                                (j * layer->stride + l)] *
                    layer->weights[o * layer->in_channels * layer->kernel_size *
                                       layer->kernel_size +
                                   c * layer->kernel_size * layer->kernel_size +
                                   k * layer->kernel_size + l];
              }
            }
          }
          sum += layer->bias[o];
          output.data[b * layer->out_channels * out_height * out_width +
                      o * out_height * out_width + i * out_width + j] = sum;
        }
      }
    }
  }
}

void conv2d_backward(Conv2D *layer, Array2D inputs, float *grad_output,
                     float *grad_input) {
  int out_height = (inputs.height - layer->kernel_size) / layer->stride + 1;
  int out_width = (inputs.width - layer->kernel_size) / layer->stride + 1;

  // Compute grad_input
  for (int b = 0; b < inputs.batch_size; b++) {
    for (int c = 0; c < layer->in_channels; c++) {
      for (int i = 0; i < inputs.height; i++) {
        for (int j = 0; j < inputs.width; j++) {
          float sum = 0.0;
          for (int o = 0; o < layer->out_channels; o++) {
            for (int k = 0; k < layer->kernel_size; k++) {
              for (int l = 0; l < layer->kernel_size; l++) {
                if (i - k * layer->stride >= 0 &&
                    i - k * layer->stride < out_height &&
                    j - l * layer->stride >= 0 &&
                    j - l * layer->stride < out_width) {
                  sum += grad_output[b * layer->out_channels * out_height *
                                         out_width +
                                     o * out_height * out_width +
                                     (i - k * layer->stride) * out_width +
                                     (j - l * layer->stride)] *
                         layer->weights[o * layer->in_channels *
                                            layer->kernel_size *
                                            layer->kernel_size +
                                        c * layer->kernel_size *
                                            layer->kernel_size +
                                        k * layer->kernel_size + l];
                }
              }
            }
          }
          grad_input[b * layer->in_channels * inputs.height * inputs.width +
                     c * inputs.height * inputs.width + i * inputs.width + j] =
              sum;
        }
      }
    }
  }

  // Compute grad_weights
  for (int o = 0; o < layer->out_channels; o++) {
    for (int c = 0; c < layer->in_channels; c++) {
      for (int k = 0; k < layer->kernel_size; k++) {
        for (int l = 0; l < layer->kernel_size; l++) {
          float sum = 0.0;
          for (int b = 0; b < inputs.batch_size; b++) {
            for (int i = 0; i < out_height; i++) {
              for (int j = 0; j < out_width; j++) {
                sum += grad_output[b * layer->out_channels * out_height *
                                       out_width +
                                   o * out_height * out_width + i * out_width +
                                   j] *
                       inputs.data[b * layer->in_channels * inputs.height *
                                       inputs.width +
                                   c * inputs.height * inputs.width +
                                   (i * layer->stride + k) * inputs.width +
                                   (j * layer->stride + l)];
              }
            }
          }
          layer->grad_weights[o * layer->in_channels * layer->kernel_size *
                                  layer->kernel_size +
                              c * layer->kernel_size * layer->kernel_size +
                              k * layer->kernel_size + l] = sum;
        }
      }
    }
  }

  // Compute grad_bias
  for (int o = 0; o < layer->out_channels; o++) {
    float sum = 0.0;
    for (int b = 0; b < inputs.batch_size; b++) {
      for (int i = 0; i < out_height; i++) {
        for (int j = 0; j < out_width; j++) {
          sum += grad_output[b * layer->out_channels * out_height * out_width +
                             o * out_height * out_width + i * out_width + j];
        }
      }
    }
    layer->grad_bias[o] = sum;
  }
}

Conv1D *conv1d_init(int in_channels, int out_channels, int kernel_size,
                    int stride) {
  Conv1D *layer = (Conv1D *)malloc(sizeof(Conv1D));
  layer->in_channels = in_channels;
  layer->out_channels = out_channels;
  layer->kernel_size = kernel_size;
  layer->stride = stride;
  layer->weights = (float *)malloc(out_channels * in_channels * kernel_size *
                                   kernel_size * sizeof(float));
  layer->bias = (float *)malloc(out_channels);

  // Initialize weights and bias with random values
  gaussian(layer->weights,
           out_channels * in_channels * kernel_size * kernel_size, RAND_MEAN,
           RAND_STD);
  gaussian(layer->bias, out_channels, RAND_MEAN, RAND_STD);

  float *grad_weights = (float *)calloc(
      out_channels * in_channels * kernel_size * kernel_size, sizeof(float));
  float *grad_bias = (float *)calloc(out_channels, sizeof(float));

  layer->forward = conv1d_forward;
  layer->backward = conv1d_backward;
  return layer;
}

// Function to create a Conv2D layer
Conv2D *conv2d_init(int in_channels, int out_channels, int kernel_size,
                    int stride) {
  Conv2D *layer = (Conv2D *)malloc(sizeof(Conv2D));
  layer->in_channels = in_channels;
  layer->out_channels = out_channels;
  layer->kernel_size = kernel_size;
  layer->stride = stride;
  layer->weights = (float *)malloc(out_channels * in_channels * kernel_size *
                                   kernel_size * sizeof(float));
  layer->bias = (float *)malloc(out_channels);

  float *grad_weights = (float *)calloc(
      out_channels * in_channels * kernel_size * kernel_size, sizeof(float));
  float *grad_bias = (float *)calloc(out_channels, sizeof(float));

  // Initialize weights and bias with random values
  gaussian(layer->weights,
           out_channels * in_channels * kernel_size * kernel_size, RAND_MEAN,
           RAND_STD);
  gaussian(layer->bias, out_channels, RAND_MEAN, RAND_STD);

  layer->forward = conv2d_forward;
  layer->backward = conv2d_backward;
  return layer;
}

// Function to destroy a Conv2D layer
void free_conv2d(Conv2D *layer) {
  free(layer->weights);
  free(layer->bias);
  free(layer->grad_weights);
  free(layer->grad_bias);
  free(layer);
}

// Function to destroy a Conv1D layer
void free_conv1d(Conv1D *layer) {
  free(layer->weights);
  free(layer->bias);
  free(layer->grad_weights);
  free(layer->grad_bias);
  free(layer);
}
int main() {
  // Example usage
  // ------------------- TESTING Conv2D ----------------------------------
  int batch_size = 2;
  int seq_len = 10;
  int in_channels = 3;
  int out_channels = 4;
  int kernel_size = 3;
  int stride = 2;

  Array1D *_input = allocate_array1d(batch_size, in_channels, seq_len);
  gaussian(_input->data, batch_size * seq_len * in_channels, 1.0, 1.);

  Array1D *_output = allocate_array1d(batch_size, out_channels,
                                      ((seq_len - kernel_size) / stride + 1));
  gaussian(_output->data, batch_size * seq_len * in_channels, 1.0, 1.);

  Conv1D *_layer = conv1d_init(in_channels, out_channels, kernel_size, stride);
  _layer->forward(_layer, *_input, *_output);
  _layer->backward(_layer, *_input, _output->grad, _input->grad);

  // Print output and gradients
  for (int i = 0;
       i < batch_size * out_channels * ((seq_len - kernel_size) / stride + 1);
       i++) {
    printf("%f ", _output->grad[i]);
  }
  printf("\n");
  for (int i = 0; i < batch_size * in_channels * seq_len; i++) {
    printf("%f ", _input->grad[i]);
  }
  printf("\n");
  for (int i = 0; i < out_channels * in_channels * kernel_size; i++) {
    printf("%f ", _layer->grad_weights[i]);
  }
  printf("\n");
  for (int i = 0; i < out_channels; i++) {
    printf("%f ", _layer->grad_bias[i]);
  }
  printf("\n");

  free(_input);
  free(_output);
  free(_input->grad);
  free(_output->grad);
  free_conv1d(_layer);
  // ------------------- TESTING Conv2D ----------------------------------
  batch_size = 2;
  int height = 10;
  int width = 10;
  in_channels = 3;
  out_channels = 4;
  kernel_size = 3;
  stride = 2;

  Array2D *input = allocate_array2d(batch_size, in_channels, height, width);
  gaussian(input->data, batch_size * in_channels * height * width, 10.0, 1.);

  Array2D *output = allocate_array2d(batch_size, out_channels,
                                     ((height - kernel_size) / stride + 1),
                                     ((width - kernel_size) / stride + 1));

  // Initialize inputs and grad_output
  for (int i = 0; i < batch_size * in_channels * height * width; i++) {
    input->data[i] = (float)i;
  }
  for (int i = 0;
       i < batch_size * out_channels * ((height - kernel_size) / stride + 1) *
               ((width - kernel_size) / stride + 1);
       i++) {
    output->grad[i] = (float)i;
  }

  Conv2D *layer = conv2d_init(in_channels, out_channels, kernel_size, stride);
  layer->forward(layer, *input, *output);
  layer->backward(layer, *input, output->grad, input->grad);

  // Print output and gradients
  for (int i = 0;
       i < batch_size * out_channels * ((height - kernel_size) / stride + 1) *
               ((width - kernel_size) / stride + 1);
       i++) {
    printf("%f ", output->data[i]);
  }
  printf("\n");
  for (int i = 0; i < batch_size * in_channels * height * width; i++) {
    printf("%f ", input->grad[i]);
  }
  printf("\n");
  for (int i = 0; i < out_channels * in_channels * kernel_size * kernel_size;
       i++) {
    printf("%f ", layer->grad_weights[i]);
  }
  printf("\n");
  for (int i = 0; i < out_channels; i++) {
    printf("%f ", layer->grad_bias[i]);
  }
  printf("\n");

  free(input->data);
  free(output->data);
  free(output->grad);
  free(input->grad);
  free(input);
  free(output);
  free_conv2d(layer);

  return 0;
}

/*
// Function to perform the forward pass of the Conv2D layer
void conv2d_forward_prev(Conv2D *layer, Array2D input, float *output) {
  int B = input.height;
  int H_out = input.width - layer.kernel_size + 1;
  int W_out = input.width - layer->kernel_size + 1;
  // float *output = (float *)malloc(B * layer->C_out * H_out * W_out *
  // sizeof(float)); Array2D *output = allocate_array2d(B, layer->C_out *
H_out
  // * W_out);

  for (int N_i = 0; N_i < B; N_i++) {
    for (int C_out_j = 0; C_out_j < layer->out_channels; C_out_j++) {
      Array2D *cross_corr_sum = allocate_array2d(H_out, W_out);
      for (int k = 0; k < layer->in_channels; k++) {

        float *c;
        cross_correlation(
            &input.data[N_i * input.width * input.width +
                        k * input.width * input.width],
            &layer->weights[C_out_j * layer->in_channels * layer->kernel_size *
                               layer->kernel_size +
                           k * layer->kernel_size * layer->kernel_size],
            (int[]){H_out, W_out}, (int[]){N_i, k}, (int[]){C_out_j, k}, c);

        for (int i = 0; i < H_out; i++) {
          for (int j = 0; j < W_out; j++) {
            cross_corr_sum->data[i * W_out + j] += c->data[i * W_out + j];
          }
        }
        free(c);
      }
      for (int i = 0; i < H_out; i++) {
        for (int j = 0; j < W_out; j++) {
          output[N_i * layer->out_channels * H_out * W_out +
                 C_out_j * H_out * W_out + i * W_out + j] =
              layer->bias[C_out_j] + cross_corr_sum->data[i * W_out + j];
        }
      }
      deallocate_array2d(cross_corr_sum);
    }
  }
  return;
}
*/
