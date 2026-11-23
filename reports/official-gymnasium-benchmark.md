# CartPole Results

I trained and tested three different approaches to the CartPole problem to see how well each one could keep the pole balanced.

I used Gymnasium's `CartPole-v1`, where the highest possible reward in an episode is 500. For the neural network, I trained for 100 epochs with 40 episodes per epoch, then tested each agent over 100 episodes.

## How they performed

| Agent | Average reward | Episodes with a reward of 475 or higher |
| --- | ---: | ---: |
| Random | 21.7 / 500 | 0 / 100 |
| Physics-based | 500.0 / 500 | 100 / 100 |
| Neural network | 467.1 / 500 | 76 / 100 |

The neural network did much better than choosing random actions. It balanced the pole for almost the full episode on average, although it still struggled in some runs.

What surprised me was that the physics-based approach performed the best. It managed to keep the pole balanced for all 500 steps in every test episode. That showed me that training a model isn't always necessary when a straightforward rule already works well.

## Training the neural network

I used a neural network with two hidden layers of 64 neurons each. The model learns from the better episodes, and I gradually reduced the amount of random exploration during training.

During training, the average reward in the final batch reached **108.3**. When I tested the trained model without random exploration, its average reward was **467.1**. The difference makes sense because it was still taking exploratory actions during training.

## Files from the experiment

I ran the experiment with GitHub Actions. The [completed run](https://github.com/koufosa/cartpole-rl-benchmark/actions/runs/45241257740) includes the output files, and the [results download](https://github.com/koufosa/cartpole-rl-benchmark/actions/runs/45241257740/artifacts/14852625271) has the training graph, evaluation graph, CSV results, and saved model.

## What I'd try next

I'd like to train the model with a few different random seeds, experiment with how quickly I reduce exploration, and see whether a different training approach could perform more consistently.
