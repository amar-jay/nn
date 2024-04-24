#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BATCH_SIZE 4

typedef struct Layer {
  uint T;
  uint fan_in;
  uint fan_out;

  float *weight; // weights of the layer (B, fan_out, fan_in)
  float *bias;   // biases of the layer (fan_out, )
                 //  float *out;    // output of the layer (B, T, fan_out)
  void (*forward)(struct Layer l, float *inp,
                  float *out); // params: (Layer, inp, out)
  void (*_backward)(struct Layer l, float *inp,
                    float *out); // params: (Layer, inp, out)

  float *inp; // input of the layer (B, T, fan_in)

  float *dinp;    // gradient of the input (B, T, fan_in)
  float *dweight; // gradient of the weights (fan_out, fan_in)
  float *dbias;   // gradient of the biases (fan_out, )
} Layer;

#define PI 3.14159265

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

void matmul_forward(float *out, float *inp, float *weight, float *bias, int B,
                    int T, int C, int OC) {
  // most of the running time is spent here and in matmul_backward
  // inp is (B,T,C), weight is (fan_out, fan_in), bias is (fan_out)
  // out will be (B,T,fan_out)

#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float *out_bt = out + b * T * OC + t * OC;
      float *inp_bt = inp + b * T * C + t * C;
      for (int o = 0; o < OC; o++) {
        float val = (bias != NULL) ? bias[o] : 0.0f;
        float *wrow = weight + o * C;
        for (int i = 0; i < C; i++) {
          val += inp_bt[i] * wrow[i];
        }
        out_bt[o] = val;
      }
    }
  }
}

void matmul_backward(float *dinp, float *dweight, float *dbias, float *dout,
                     float *inp, float *weight, int B, int T, int C, int OC) {
// most of the running time is spent here and in matmul_forward
// this backward could be done in a single "round" of loops
// but that doesn't afford an efficient parallelization strategy

// backward into inp first, parallelize over B,T
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float *dout_bt = dout + b * T * OC + t * OC;
      float *dinp_bt = dinp + b * T * C + t * C;
      for (int o = 0; o < OC; o++) {
        float *wrow = weight + o * C;
        float d = dout_bt[o];
        for (int i = 0; i < C; i++) {
          dinp_bt[i] += wrow[i] * d;
        }
      }
    }
  }
}

void linear_forward(Layer l, float *x, float *xout) {
  // inp is (B,T,C), weight is (fan_out, fan_in), bias is (fan_out)
  // out will be (B,T,fan_out)
  matmul_forward(xout, x, l.weight, l.bias, BATCH_SIZE, l.T, l.fan_in,
                 l.fan_out);
}

/*
TODO: Implement this function later, after understanding backpropagation for
multidimensional arrays void linear_backward(Layer l, float *dout) { if (l.dinp
== NULL) {

    printf("Error: backward called without allocating gradients\n");
    exit(1);
  }

  matmul_backward(l.dinp, l.dweight, l.dbias, dout, l.inp, l.weight,
                  BATCH_SIZE, l.T, l.fan_in, l.fan_out);
}
*/

Layer *linear_init(uint fan_in, uint fan_out, bool bias) {
  Layer *layer = malloc(sizeof(Layer));
  layer->weight = calloc(BATCH_SIZE * fan_out * fan_in, sizeof(float));

  if (bias) {
    layer->bias = calloc(fan_out, sizeof(float));
  } else {
    layer->bias = NULL;
  }

  layer->fan_in = fan_in;
  layer->fan_out = fan_out;
  layer->forward = linear_forward;

  gaussian(layer->weight, BATCH_SIZE * fan_in * fan_out, 0, 1.);
  gaussian(layer->bias, fan_out, 0, 1.);

  return layer;
}

void free_linear(Layer *layer) {
  free(layer->weight);
  free(layer->bias);
  free(layer->inp);
  free(layer->dinp);
  free(layer->dweight);
  free(layer->dbias);
  free(layer);
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float *out, float *inp, int N) {
  // (approximate) GeLU elementwise non-linearity in the MLP block of
  // Transformer
  for (int i = 0; i < N; i++) {
    float x = inp[i];
    float cube = 0.044715f * x * x * x;
    out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
  }
}

int main() {
  srand(time(NULL));
  Layer *layer1 = linear_init(10, 15, true);
  Layer *layer2 = linear_init(15, 10, true);

  float *inp = calloc(BATCH_SIZE * 10, sizeof(float));
  gaussian(inp, BATCH_SIZE * 10, 0, 1.);

  float *hid = calloc(BATCH_SIZE * 15, sizeof(float));
  layer1->forward(*layer1, inp, hid);

  gelu_forward(hid, hid, BATCH_SIZE * 15);

  float *out = calloc(BATCH_SIZE * 10, sizeof(float));
  layer2->forward(*layer2, inp, out);

  free(inp);
  free(hid);
  free(out);
  free_linear(layer1);
  free_linear(layer2);

  return 0;
}
