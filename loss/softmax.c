#include <math.h>

// softmax function
// it is an element-wise operation of exp(x_i) / sum(exp(x))
void softmax(float *in, float *out, int size) {

  float sum = 0;

  for (int i = 0; i < size; i++) {
    sum += exp(in[i]);
  }

  for (int i = 0; i < size; i++) {
    out[i] = exp(in[i]) / sum;
  }
}

void softmax_backward(float *grads, float *logits, float *target, int size) {
  for (int i = 0; i < size; i++) {
    grads[i] = logits[i] - target[i];
  }
}

// cross entropy loss function
// formula is -sum(target * log(logits))
float cross_entropy(float *logits, float *target, int size) {

  float loss = 0;

  for (int i = 0; i < size; i++) {
    loss += target[i] * log(logits[i]);
  }

  return -loss;
}

void cross_entropy_backward(float *grads, float *logits, float *target,
                            int size) {
  for (int i = 0; i < size; i++) {
    grads[i] = -target[i] / logits[i];
  }
}
