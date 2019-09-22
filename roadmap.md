# Project roadmap

### 1. Attempt to achive better results using classic LSTM architecture

- prepare simpler dataset suitable for quicker learning & evaluating diffrent architectures/hyperparameters
- experiment with altered data representation
  - embedding sparse note vectors
  - alternative time representation ???
- experiment with altered LSTM shape / architecture + hyperparameters
  - using Dropout / Recurrent Dropout
  - stacked LSTM
  - bidirectional LSTM
- potentially test best approach on Beethoven dataset

### 2. Develop recurrent Generative Adversial Model (GAN)

- investigate GAN learning model
- study available research papers about recurrent GANs
- design / develop recurrent GAN model using TF.Keras
  - evaluate the model
  - attempt to train it using simple dataset
  - if above steps succeed, use the Beethoven dataset
