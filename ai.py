import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


ALPHA = 0.9
LR = 0.005
SAVE_PATH = './save/checkpoint'
RANGE_SAVE = 100

class Network(keras.Model):
	""" Subclassing classe pour un réseau, j'aime pas trop les Séquentiels et les compiles et fit vu que j'ai commencé par Pytorch"""

	def __init__(self, nb_outputs, epsilon):
		super(Network, self).__init__()

		self.nb_outputs = nb_outputs
		self.epsilon = epsilon
		self.dense1 = keras.layers.Dense(30, activation="linear", dtype=tf.float64) # J'ai cru bon de pas anéantir mes coords en ne mettant pas une relu qui va zapper les 0
		self.dense2 = keras.layers.Dense(30, activation="tanh", dtype=tf.float64) # après je savais pas trop quoi mettre, puis j'ai vu la tanh je sais pas ou du coup j'ai tenté
		self.dense3 = keras.layers.Dense(30, activation="tanh", dtype=tf.float64)
		self.dense4 = keras.layers.Dense(nb_outputs, activation="softmax", dtype=tf.float64) # le bon vieux softmax qui fait plaisir

	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.dense2(x)
		x = self.dense3(x)
		x = self.dense4(x)

		return x

	def select_action(self, state):
		if random.random() > self.epsilon: # l'implémentation basique de epislon greedy, je l'ai peut-être fait à l'envers mais bon
			return np.argmax(self.predict(state)) # on retourne la plus grande valeur du predict qui sera aussi la direction du cart soit 0 ou 1 pour respectivement gaucge et droite
		else:
			print('ALEATOIRE')
			return float(random.randrange(self.nb_outputs)) # hop hop on pique un chiffre aléatoire entre 0 et 1


class ReplayMemory:
	def __init__(self, capacity):
		self.memory = []
		self.capacity = capacity

	def push(self, transition): # les transitions comportent [state, next_state, action, reward]
		if len(self.memory) >= self.capacity:
			del self.memory[0]
		self.memory.append(transition)

	def sample(self, batch_size):
		sample = random.sample(self.memory, batch_size) # on sélectionne un lot de manière aléatoire dans la mémoire
		batch = list(zip(*sample)) # on rassemble les state avec les state, les actions avec les actions....
		batch[0] = tf.constant(batch[0], dtype=tf.float64)  # on met le tout sous tensor pour pouvoir profiter du mode graph et parce que ça m'évite de le faire à gauche à droite
		batch[1] = tf.constant(batch[1], dtype=tf.float64)
		batch[2] = tf.constant(batch[2], dtype=tf.float64)
		batch[3] = tf.constant(batch[3], dtype=tf.float64)

		return batch

	def size(self):
		return len(self.memory)


class DQN:
	def __init__(self, nb_outputs, size_memory, batch_size, epsilon):
		self.alpha = tf.constant(ALPHA, dtype=tf.float64)
		self.batch_size = batch_size # le nombre de transition par lot
		self.model = Network(nb_outputs, epsilon)
		self.loss = keras.losses.MeanSquaredError() # je voulais la huber losss mais j'avais la flemme de l'implémenter
		self.optimizer = keras.optimizers.Adam(lr=LR) # le bon vieux Adam, je crois que c'est un merci google pour celui ci
		self.memory = ReplayMemory(size_memory)
		self.list_step_loss = [[], []] # list qui servira à faire un graphique de l'erreur en fonction des étapes
		self.counter_update = 0 # un compteur pour le graphique, donc les étapes
		self.counter_save = 0 # un compteur pour l'intervalle de sauvegarde
		self.last_state = [[0.0, 0.0, 0.0, 0.0]]
		self.last_action = 0
		self.last_reward = 0
		self.restore() # on restaure une save si disponible

	def learn(self, batch_state, batch_next_state, batch_action, batch_reward):
		print('ENTRAINEMENT')
		try:
			with tf.GradientTape() as tape:
				q_actions = tf.gather(self.model(batch_state)[0], tf.cast(batch_action, tf.int32), axis=0) # on récupère les actions choisi grâce aux states
				q_next_actions_max = tf.reduce_max(self.model(batch_next_state), axis=1) # on récupère les max des des q_action de next_state
				targets = batch_reward + self.alpha * q_next_actions_max # Eq de Bellman
				loss = self.loss(targets, q_actions) # on calcule l'erreur
				gradients = tape.gradient(loss, self.model.trainable_variables) # on cacule le gradient, enfin
				self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables)) # on actualise les poids avec Adam
				if self.counter_save == 100:
					self.list_step_loss[0].append(self.counter_update)
					self.list_step_loss[1].append(loss.numpy())
				print("l'erreur est de {}".format(loss))
		except:
			print('Erreur dans learn')

	def update(self, next_signals, reward):
		next_state = tf.constant(next_signals, dtype=tf.float64) # on récupère les nouveaux states
		action = self.model.select_action(next_state)  # on sélectionne l'action
		self.memory.push([self.last_state[0], next_signals[0], self.last_action, reward]) # on nourrit la memoire
		if self.memory.size() > self.batch_size:
			batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(self.batch_size)
			self.learn(batch_state, batch_next_state, batch_action, batch_reward) # si il y a assez d'éléments dans la mémoire on apprend
			if self.counter_save == 100:
				self.checkpoint()
				self.counter_save = 0
		self.counter_update += 1
		self.counter_save += 1
		self.last_state = next_signals # on actualise les données utiles pour l'apprentissage
		self.last_action = action
		self.last_reward = reward
		return action

	def draw_graph(self):
		plt.title("Evolution de l'errreur en fonction des updates")
		plt.xlabel('Etapes')
		plt.ylabel('Erreur')
		plt.plot(self.list_step_loss[0], self.list_step_loss[1])
		plt.show()

	def checkpoint(self):
		self.model.save_weights(SAVE_PATH)
		self.draw_graph()
		print('Les poids du modèle ont été sauvegardés')

	def restore(self):
		try:
			with open(SAVE_PATH):
				self.model.load_weights(SAVE_PATH)
				print('Les poids du modèle ont été restaurés')
		except IOError:
			print('Pas de sauvegarde trouvé')
