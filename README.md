# Assignment *number, 4*

- name: Matthew Pereira Wilson
- student ID: 20644035

## Dependencies

- os
- json
- time
- numpy
- argparse
- matplotlib
- torch

## Running `main.py`

To run `main.py`, use

```sh
python3 main.py --input_path=data/even_mnist.csv --params_path=data/params.json -o=results_dir -n=100 -v=2 -cuda=1
```

Expect a runtime of >1min. This assignment was written while consulting: https://github.com/pytorch/examples/blob/master/vae/main.py

## Hyper-parameter `json` file

A sample `params.json` file is provided in `data/`. It contains,

- n_epoch      : number of epochs to train the model
- n_test       : number of datapoints set aside for cross-validation
- lr           : learning rate of the neural network
- n_epoch_v    : interval at which loss is printed
- n_mini_match : size of mini batch

## Outputs

This script outputs into directory results_dir. It plots the loss function from the NN training (`loss.pdf`), and N samples of generated numbers (`n.pdf`).
