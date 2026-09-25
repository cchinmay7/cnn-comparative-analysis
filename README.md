# CNN vs. DNN: A Comparative Analysis for Image Classification

A study comparing convolutional neural networks (CNNs) with fully connected networks (DNNs) for image classification on four datasets of increasing difficulty, and measuring how width (512 vs. 1024 hidden units) and training length (5 vs. 50 epochs) affect accuracy. Conducted in Summer 2023.

📄 **[Read the paper](paper.pdf)**

## Setup

- **Datasets:** MNIST, EMNIST (balanced), Fashion-MNIST, CIFAR-10
- **Architectures:** DNN (flatten → dense → softmax) and CNN (two conv layers + max pooling → dense → softmax)
- **Grid:** {DNN, CNN} × {512, 1024 hidden units} × {5, 50 epochs}
- **Framework:** TensorFlow / Keras, RMSprop, categorical cross-entropy

## Results

Best test accuracy for each architecture:

| Dataset | Best DNN | Best CNN | CNN gain |
| --- | --- | --- | --- |
| MNIST | 98.45% | 99.30% | +0.85 pts |
| Fashion-MNIST | 89.44% | 91.13% | +1.69 pts |
| EMNIST | 84.28% | 87.14% | +2.86 pts |
| CIFAR-10 | 49.88% | 69.01% | **+19.13 pts** |

**Findings**

- CNNs win on every dataset, and the gap grows with visual complexity. On near-solved MNIST the difference is under a point, while on natural-image CIFAR-10 it is nearly 20.
- Longer training helps on Fashion-MNIST and CIFAR-10 but slightly *hurts* on EMNIST (e.g. CNN-512: 87.14% → 85.73%), a sign of overfitting without regularisation.
- Doubling width from 512 to 1024 units gives small, inconsistent gains compared with changing the architecture.

## Repository

```text
experiments/
  MNIST.ipynb, Extended Mnist.ipynb, Fashion MNIST.ipynb, CIFAR-10.ipynb   training runs
  Visualization 1.ipynb, Visualization 2.ipynb                             result charts
paper.pdf                                                                  full write-up
```

## Running

```bash
pip install tensorflow extra-keras-datasets matplotlib numpy jupyter
```

Each notebook in `experiments/` downloads its dataset and trains independently.
