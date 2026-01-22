#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef RAND_MEAN
#define RAND_MEAN 0.0
#endif

#ifndef RAND_STD
#define RAND_STD 1.0
#endif
/*
typedef struct {
  int batch_size;
  int seq_len;
  int *data;
  float *grad;
} EmbeddingArray; // is there a way to do this that doesn't involve copying the
                  // repetition. since embedding holds only int not floats

*/

typedef struct {
  int batch_size;
  int seq_len;
  float *data;
  float *grad;
} Array1D;

typedef struct Embedding {
  int num_embeddings;
  int embedding_dim;
  float *weights;
  float *grad_weights;

  void (*forward)(struct Embedding *layer, int *inputs, Array1D output);
  void (*backward)(struct Embedding *layer, int *inputs, float *grad_output,
                   int batch_size, int seq_len);
} Embedding;

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

// Forward pass
// hmm, rather than passing in an embedding layer, we could pass in the pointer.
void embedding_forward(Embedding *layer, int *inputs, Array1D output) {
  for (int b = 0; b < output.batch_size; b++) {
    for (int i = 0; i < output.seq_len; i++) {
      int idx = inputs[b * output.seq_len + i];
      for (int j = 0; j < layer->embedding_dim; j++) {
        output.data[b * output.seq_len * layer->embedding_dim +
                    i * layer->embedding_dim + j] =
            layer->weights[idx * layer->embedding_dim + j];
      }
    }
  }
}

// Backward pass
void embedding_backward(Embedding *layer, int *inputs, float *grad_output,
                        int batch_size, int seq_len) {
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < seq_len; i++) {
      int idx = inputs[b * seq_len + i];
      for (int j = 0; j < layer->embedding_dim; j++) {
        layer->grad_weights[idx * layer->embedding_dim + j] +=
            grad_output[b * seq_len * layer->embedding_dim +
                        i * layer->embedding_dim + j];
      }
    }
  }
}

Embedding *embedding_init(int num_embeddings, int embedding_dim) {
  Embedding *layer = (Embedding *)malloc(sizeof(Embedding));
  layer->num_embeddings = num_embeddings;
  layer->embedding_dim = embedding_dim;
  layer->weights =
      (float *)malloc(num_embeddings * embedding_dim * sizeof(float));

  layer->grad_weights =
      (float *)malloc(num_embeddings * embedding_dim * sizeof(float));
  layer->forward = embedding_forward;
  layer->backward = embedding_backward;
  return layer;
}

void free_embedding(Embedding *layer) {
  free(layer->weights);
  free(layer->grad_weights);
  free(layer);
}

Array1D *allocate_array1d(int batch_size, int channel, int seq_len) {
  Array1D *arr = (Array1D *)malloc(sizeof(Array1D));
  arr->batch_size = batch_size;
  arr->seq_len = seq_len;
  arr->data = (float *)calloc(batch_size * channel * seq_len, sizeof(float));
  arr->grad = (float *)calloc(batch_size * channel * seq_len, sizeof(float));
  return arr;
}

int main() {
  // Example usage
  int batch_size = 2;
  int seq_len = 5;
  int num_embeddings = 10;
  int embedding_dim = 8;

  int *inputs = (int *)malloc(batch_size * seq_len * sizeof(int));
  Array1D *output = allocate_array1d(batch_size, embedding_dim, seq_len);
  float *grad_output =
      (float *)malloc(batch_size * seq_len * embedding_dim * sizeof(float));

  // Initialize inputs and grad_output
  for (int i = 0; i < batch_size * seq_len; i++) {
    inputs[i] = i % num_embeddings; // a simple positional encoding
  }

  Embedding *layer = embedding_init(num_embeddings, embedding_dim);

  embedding_forward(layer, inputs, *output);
  embedding_backward(layer, inputs, grad_output, batch_size, seq_len);

  // Print output and gradients
  for (int i = 0; i < batch_size * seq_len * embedding_dim; i++) {
    printf("%f ", output->data[i]);
  }
  printf("\n");
  for (int i = 0; i < num_embeddings * embedding_dim; i++) {
    printf("%f ", layer->grad_weights[i]);
  }
  printf("\n");

  free(inputs);
  free(output);
  free(grad_output);
  free_embedding(layer);
  return 0;
}
