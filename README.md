# PERCEIVER: General Perception with Iterative Attention

# Scheme of the Perceiver model
<figure>
  <img src="image/architecture.png" alt="Perceiver Architecture">
  <figcaption>Perceiver Architecture</figcaption>
</figure>

<figure>
    <img src="image/full_architecture.gif" alt="Perceiver Architecture">
    <figcaption>How the Perceiver model works</figcaption>
</figure>

This repository contains the implementations in PyTorch of the Perceiver model from the paper [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206).
I try to implement the model as close as possible to the original paper. 
The model is trained on the Modelnet40 dataset, and the training script is provided in the repository.

# Structure of the repository
- datasets: contains the folder of the dataset used in the experiments, in this case Modelnet40
- image: contains the schematic representation of the Perceiver model
- paper: all papers used in this project
- src: contains the implementation of the Perceiver model
- train: contains the training script of the Perceiver model
- test: contains the unittest of the Perceiver model

