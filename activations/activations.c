#include <math.h>

/**
 * Well known activation functions
 * gelu: Gaussian Error Linear Unit
 * relu: Rectified Linear Unit
 * sigmoid: Sigmoid
 * @param in: input array
 * @param out: output array
 * @param size: size of the input array
 *
 */

// gelu activation function
// formula: out[i] = x * 0.5 * (1.0 + tanh(sqrt(2 / M_PI) * (x + 0.044715 *
// pow(x, 3))))
void gelu(float *in, float *out, int size) {

  for (int i = 0; i < size; i++) {
    float x = in[i];
    float cdf =
        0.5 * (1.0 + tanh((sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3)))));
    out[i] = x * cdf;
  }
}

// gelu activation function
// formula: out[i] = in[i] > 0 ? in[i] : 0
void relu(float *in, float *out, int size) {
  for (int i = 0; i < size; i++) {
    out[i] = in[i] > 0 ? in[i] : 0;
  }
}

// sigmoid activation function
// formula: out[i] = 1 / (1 + exp(-in[i]))
void sigmoid(float *in, float *out, int size) {
  for (int i = 0; i < size; i++) {
    out[i] = 1 / (1 + exp(-in[i]));
  }
}
