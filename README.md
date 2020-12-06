# README for G24

Our project uses synthetic data from OpenAI Gym (https://gym.openai.com/envs/CartPole-v1/) and from a custom car racing environment. We are using sklearn and PyTorch for our models.

## Files

- ./cartpole-q-learning-nn.ipynb -> training and results for OpenAI CartPole using Neural Networks
- ./cartpole-q-learning-tree.ipynb -> training and results for OpenAI CartPole using LightGBM
- ./custom-car-q-learning-nn.ipynb -> training and results for custom car environment using Neural Networks
- ./custom-car-q-learning-tree.ipynb -> training and results for custom car environment using LightGBM

## Instructions
- The order in which to run the files does not matter, but we recommend to follow the order listed above
- Training takes between 20 minutes and 1.5 hours for each notebook
- GPU not required
- Each notebook is independent (leads to some code repeat, but allows to train in parallel and was generally more convenient)

## Reproducibility
- See ./environment.yaml for the conda environment used
- Random seeds were not set because environments are inherently random (random actions are taken to build memory). Setting a random seed would not have helped reproducibility in any way.
- Because of the inherent stochasticity, not all runs will look exactly the same, but overall trends should persist.

### References
- General inspiration: https://towardsdatascience.com/reinforcement-learning-q-learning-with-decision-trees-ecb1215d9131
