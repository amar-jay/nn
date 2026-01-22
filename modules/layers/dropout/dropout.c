#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Dropout layer
typedef struct Dropout {
  float p;
  void (*forward)(struct Dropout, float *, int, float *, unsigned int *);

  void (*backward)(struct Dropout layer, float *grad_output, int size,
                   unsigned int *mask, float *grad_input);
} Dropout;

void dropout_forward(Dropout layer, float *inputs, int size, float *output,
                     unsigned int *mask) {
  for (int i = 0; i < size; i++) {
    if (((float)rand() / RAND_MAX) < layer.p) {
      output[i] = 0.0;
      mask[i] = 0;
    } else {
      output[i] = inputs[i];
      mask[i] = 1;
    }
  }
}

// though layer is not used, it is kept for consistency.
void dropout_backward(Dropout layer, float *grad_output, int size,
                      unsigned int *mask, float *grad_input) {
  for (int i = 0; i < size; i++) {
    if (mask[i] == 1) {
      grad_input[i] = grad_output[i];
    } else {
      grad_input[i] = 0.0;
    }
  }
}

Dropout dropout_init(float p) {
  Dropout layer;
  layer.p = p;
  layer.forward = dropout_forward;
  layer.backward = dropout_backward;
  return layer;
}

int main() {
  // Example usage
  int size = 10;
  float p = 0.5;

  float *inputs = (float *)malloc(size * sizeof(float));
  float *output = (float *)malloc(size * sizeof(float));
  float *grad_output = (float *)malloc(size * sizeof(float));
  float *grad_input = (float *)malloc(size * sizeof(float));
  unsigned int *mask = (unsigned int *)malloc(size * sizeof(unsigned int));

  // Initialize inputs and grad_output
  for (int i = 0; i < size; i++) {
    inputs[i] = (float)i; // linspace(0, 9, 10)
    grad_output[i] = (float)i;
  }

  Dropout layer;
  layer.p = p;

  srand(time(NULL)); // seed for random number generation

  layer.forward(layer, inputs, size, output, mask);
  layer.backward(layer, grad_output, size, mask, grad_input);

  // Print output and grad_input
  for (int i = 0; i < size; i++) {
    printf("%f ", output[i]);
  }
  printf("\n");
  for (int i = 0; i < size; i++) {
    printf("%f ", grad_input[i]);
  }
  printf("\n");

  free(inputs);
  free(output);
  free(grad_output);
  free(grad_input);
  free(mask);

  return 0;
}
