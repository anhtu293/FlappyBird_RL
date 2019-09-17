from ple import PLE
from ple.games.flappybird import FlappyBird
import numpy as np
from skimage.transform import resize
from skimage import img_as_float
from skimage.color import rgb2gray
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
import itertools
import random

MEMORY_BUFFER = []
MEMORY_BUFFER_SIZE = 100000
NUMBER_EPISODES = 10000
BATCH_SIZE = 32

def nth_root(num, n):
	return(n**(1/num))

def build_model(env):
	#dim action space
	nA = len(env.getActionSet())
	
	input_shape = (64, 64, 3)
	input_nn = Input(shape = input_shape)
	conv3_1 = Conv2D(16, kernel_size = 3,strides = 1, padding = 'same', activation = 'relu')(input_nn)
	conv3_2 = Conv2D(32, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(conv3_1)
	flatten1 = Flatten()(conv3_2)
	fc1 = Dense(64, activation = 'relu')(flatten1)
	fc2 = Dense(nA, activation = 'linear')(fc1)

	model = Model(inputs = input_nn, outputs= fc2)

	optimizer = optimizers.Adam(0.0001)
	model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['mse'])
	return model

def approximation(model, state):
	return model.predict(state, verbose = 0)

def epsilon_greedy_policy(model, state, nA, epsilon):
	A = np.ones(nA)*epsilon/nA
	best_action = np.argmax(approximation(model, state))
	A[best_action] += 1 - epsilon
	action = np.random.choice(range(nA), p = A)
	return action

def updateModel(model, state, updatedQ):
	return model.fit(state, updatedQ)

def DeepQLearning(epsilon = 1, discount = 0.99):
	game = FlappyBird()

	rewards = {
		'positive' : 1,
		'stick' : 0,
		'loss' : 0	
	}

	env = PLE(game, fps = 30, display_screen = False, reward_values = rewards)
	env.init()

	model = build_model(env)

	#parameters
	nA = len(env.getActionSet())
	final_epsilon = 0.001
	epsilon_decay = nth_root(NUMBER_EPISODES, final_epsilon/epsilon)

	print("=========== Start Training ===========\n")

	for i in range(1, NUMBER_EPISODES):
		epsilon = epsilon*epsilon_decay
		score = 0
		avg_score = []
		if (i % 1000 == 0):
			avg = np.mean(np.asarray(avg_score))
			model.save_weights('episode_{}_AvgScore_{}.hdf5'.format(i, avg))
			avg_score.clear()
			print("\nEpisode_{}_AvgScore_{}.hdf5 Saved !")

		for t in itertools.count():
			#appro next action
			state = img_as_float(resize(env.getScreenRGB(), (64,64)))
			state = state.reshape((-1, 64, 64, 3))
			action = epsilon_greedy_policy(model, state, nA, epsilon)
			reward = env.act(action)
			next_state = img_as_float(resize(env.getScreenRGB(), (64,64)))
			next_state = next_state.reshape((-1, 64, 64, 3))
			score += reward
			done = env.game_over()

			avg_score.append(score)

			if not env.game_over():
				reward += discount*np.max(approximation(model, state))

			if len(MEMORY_BUFFER) == MEMORY_BUFFER_SIZE:
				MEMORY_BUFFER.clear()
			MEMORY_BUFFER.append((state, action, reward, next_state, done))

			if env.game_over():
				break

			experience_replay(env, model, discount)

		print("\nEpisode {}/{} ---- Score : {}".format(i,NUMBER_EPISODES, score))

	return model
		
def experience_replay(env, model, discount):
	batch_state = random.choices(MEMORY_BUFFER, k = BATCH_SIZE)
	for state, action, reward, next_state, done in batch_state:
		Q_update = reward
		if not done:
			Q_update += discount*np.max(approximation(model, state))
		Q_to_update = approximation(model, state)
		Q_to_update[state][action] = Q_update
		updateModel(model, state, Q_to_update)


if __name__ == '__main__':
	model = DeepQLearning()
	model.save_model('FlappyBird.h5')