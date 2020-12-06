# README for G24

Our project uses synthetic data from OpenAI Gym (https://gym.openai.com/envs/CartPole-v1/) and from a custom car racing environment. We are using sklearn and PyTorch for our models.

## Files

- ./pendulum-q-learning-nn.ipynb -> training and results for OpenAI pendulum using Neural Networks
- ./pendulum-q-learning-tree.ipynb -> training and results for OpenAI pendulum using LightGBM
- ./custom-car-q-learning-nn.ipynb -> training and results for custom car environment using Neural Networks
- ./custom-car-q-learning-tree.ipynb -> training and results for custom car environment using LightGBM

## Instructions
- The order in which to run the files does not matter, but we recommend to follow the order listed above
- Training takes between 20 minutes and 1.5 hours for each notebook
- GPU not required
- Each notebook is independent (leads to some code repeat, but allows to train in parallel and was generally more convenient)

### References
- General inspiration: https://towardsdatascience.com/reinforcement-learning-q-learning-with-decision-trees-ecb1215d9131
