#AI gets pixel inputs from the game window and makes decision based on where the ball is
"""
Needs to be debugged
tested 
Updated to save best two AI
"""

#Authors
#@BB00GIE
#@TreJohnson1




import gym
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidDerivitive(x):
    return x * (1 - x)


def Degrade(image):
    return image[::2, ::2, :]


def GrayScale(image):
    return image[:, :, 0]


def Delete_BackGround(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image


def observe(input, previous, dimension):
    Observations1 = input[35:195]
    Observations1 = Degrade(Observations1)
    Observations1 = GrayScale(Observations1)
    Observations1 = Delete_BackGround(Observations1)
    Observations1[Observations1 != 0] = 1
    Observations1 = Observations1.astype(np.float).ravel()

    if previous is not None:
        input = Observations1 - previous
    else:
        input = np.zeros(dimension)

    previous = Observations1
    return input, previous


def perturb(vector):
    vector[vector < 0] = 0
    return vector


def Nets(Matrix, weights):
    hiddenLayer = np.dot(weights['1'], Matrix)
    hiddenLayer = perturb(hiddenLayer)
    outputlayer = np.dot(hiddenLayer, weights['2'])
    outputlayer = sigmoid(outputlayer)
    return hiddenLayer, outputlayer


def Decision(probability):
    random_num = np.random.uniform()
    if random_num < probability:
        return 2
    else:
        return 3


def compute(gradient, hiddenlayer, Observations, weights):
    delta_L = gradient
    dC_dw2 = np.dot(hiddenlayer.T, delta_L).ravel()
    delta_L2 = np.outer(delta_L, weights['2'])
    delta_L2 = perturb(delta_L2)
    dC_dw1 = np.dot(delta_L2.T, Observations)
    return {
        "1": dC_dw1,
        "2": dC_dw2
    }


def new_weights(weights, expectation, dict, decay_rate, learn_rate):
    epsilon = 1e-5
    for layer_name in weights.keys():
        g = dict[layer_name]
        expectation[layer_name] = decay_rate * expectation[layer_name] + (1 - decay_rate) * g ** 2
        weights[layer_name] += (learn_rate * g) / (np.sqrt(expectation[layer_name] + epsilon)),
        dict[layer_name] = np.zeros_like(weights[layer_name])


def Dis_Rewards(rewards, gamma):
    dis_Rewards = np.zeros_like(rewards)
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        dis_Rewards[t] = running_add
    return dis_Rewards


def Dis_with_Rewards(gradient, episode, gamma):
    Discounted_rewards = Dis_Rewards(episode, gamma)
    Discounted_rewards -= np.mean(Discounted_rewards)
    Discounted_rewards /= np.std(Discounted_rewards)
    return gradient * Discounted_rewards


def main():
    env = gym.make("Pong-v0")
    observation = env.reset()

    episode_number = 0
    batch_size = 10
    gamma = 0.99
    decay_rate = 0.99
    num_hidden_layer_neurons = 200
    input_dimensions = 80 * 80
    learning_rate = 1e-4

    episode_number = 0
    reward_sum = 0
    running_reward = None
    previous = None

    weights = {
        '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
        '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
    }
    expectation = {}
    dict = {}
    for layer_name in weights.keys():
        expectation[layer_name] = np.zeros_like(weights[layer_name])
        dict[layer_name] = np.zeros_like(weights[layer_name])

    episode_hidden_layer_values, episode_obs, episode_grad, episode_rewards = [], [], [], []

    while True:
        env.render()
        processed_observations, prev_processed_observations = observe(observation, previous, input_dimensions)
        hidden_layer_values, up_probability = Nets(processed_observations, weights)

        episode_obs = np.append(episode_obs, processed_observations)
        episode_hidden_layer_values.append(hidden_layer_values)

        action = Decision(up_probability)

        # carry out the chosen action
        observation, reward, done, info = env.step(action)

        reward_sum += reward
        episode_rewards.append(reward)

        # see here: http://cs231n.github.io/neural-networks-2/#losses
        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probability
        episode_grad.append(loss_function_gradient)

        if done:  # an episode finished
            episode_number += 1

            # Combine the following values for the episode
            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_obs = np.vstack(episode_obs)
            episode_grad = np.vstack(episode_grad)
            episode_rewards = np.vstack(episode_rewards)
 
            # Tweak the gradient of the log_ps based on the discounted rewards
            episode_grad_discounted = Dis_with_Rewards(episode_grad, episode_rewards, gamma)

            gradient = compute(
                episode_grad_discounted,
                episode_hidden_layer_values,
                episode_obs,
                weights
            )

            # Sum the gradient for use when we hit the batch size
            for layer_name in gradient:
                dict[layer_name] += gradient[layer_name]

            if episode_number % batch_size == 0:
                new_weights(weights, expectation, dict, decay_rate, learning_rate)

            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []  # reset values
            observation = env.reset()  # reset env
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print
            'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
            reward_sum = 0
            prev_processed_observations = None


main()
