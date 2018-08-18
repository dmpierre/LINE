### Large Scale Information Network Embedding (LINE) - PyTorch Implementation

*For python 3 and above. This is a toy implementation and should be treated as so.*

#### Description

The [LINE algorithm](https://arxiv.org/pdf/1503.03578.pdf) was proposed in 2015 by [Jian Tang](https://github.com/tangjianpku/LINE).

This is a PyTorch implementation, which can be trained on a GPU - following your hardware. At the cost of speed, it also is trainable on CPU.

#### Usage

It is recommended to run the model within a ```virtualenv```.

Beforehand, install the required dependencies:

```
$ (env) pip install -r requirements.txt
```

Run:

```
python ./train.py -g ./data/erdosrenyi.edgelist -save ./model.pt -lossdata ./loss.pkl -epochs 10
```

Available hyperparameters are:

- ```--order```: Order 1 or 2 for the LINE algorithm.
- ```--negativepower```: Power used for raising the nodes out-degree distribution.
- ```--negsamplesize```: number of negative examples used. Defaults to 5.
- ```--batchsize```: batchsize during training.
- ```--epochs```: Number of epochs for training.
- ```--learning_rate```: Learning rate aggressiveness.
