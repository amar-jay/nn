Great day! I think I have a rough idea on optimizers work.

- I have no idea why, RMSProp keeps giving absurd results though I followed it line-by-line, that of GPT is way more inefficient
- Bcs I'm too exhausted, I told Llama to generate Adam Optimizer and it works fairly well, Just overlook the some deprecation warnings when executing

```
Model, Accuracy, Epochs, % Difference
StochasticGradientDescent, 100.000,  1440,   18.25
SGDWithMomentum     , 100.000,   300,   12.17
SGDWithNestrov      ,  40.000,   180,   21.44
RMSProp             ,  10.000,  4320,  328.61
Adam                , 100.000,   180,   10.04

```
