import random

from collections import deque

import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.models import load_model, clone_model
from keras.layers import Dense
from keras.optimizers import Adam

from keras.layers import Dense, Activation, Flatten, Conv2D, Conv1D, MaxPooling2D, LSTM, ConvLSTM2D
from datetime import datetime


def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning

    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))


class Agent:
    """ Stock Trading Bot """

    def __init__(self, window_size, strategy="t-dqn", reset_every=1000, pretrained=False, model_name=datetime.now().strftime("20%y_%m_%d__%H_%M"),
                 n_feachs=14):
        self.strategy = strategy

        # agent config
        self.state_size = (window_size, n_feachs)  # normalized previous days
        self.action_size = 3  # [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=10000)
        self.first_iter = True

        # model config
        self.model_name = model_name
        self.gamma = 0.95  # affinity for long term reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.05
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        self.optimizer = Adam(lr=self.learning_rate)

        if pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.model = self._model()

        # strategy config
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every

            # target network
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def _model(self):
        """Creates the model
        """
        model = Sequential()

        # with tf.device("/gpu:0"):
        #     model = tf.keras.models.Sequential([
        #         tf.keras.layers.Flatten(input_shape=self.state_size),
        #         tf.keras.layers.Dense(1024, activation='relu'),
        #         tf.keras.layers.Dense(256, activation='relu'),
        #         tf.keras.layers.Dense(256, activation='relu'),
        #         tf.keras.layers.Dense(128, activation='relu'),
        #         tf.keras.layers.Dense(3),
        #     ])
        #
        # ---------------------------------------------------------------------------------
        # Conv Layers
        # model.add(Conv1D(32, 8, strides=4, padding='same', input_shape=self.state_size))
        # model.add(Activation('relu'))
        #
        # model.add(Conv1D(64, 4, strides=2, padding='same'))
        # model.add(Activation('relu'))
        #
        # model.add(Conv1D(64, 3, strides=1, padding='same'))
        # model.add(Activation('relu'))
        # model.add(Flatten())
        #
        # # FC Layers
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(3, activation='softmax'))
        #
        # ------------------------------------------------------------------------
        model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=self.state_size))
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model

    def remember(self, state, action, reward, next_state, done):
        """Adds relevant data to memory
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval=False):
        """Take action from given possible set of actions
        """
        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1  # make a definite buy on the first iter
        #print("act" + str(state.shape))
        state = state.reshape(1, 50, 14)
        action_probs = self.model.predict(state, verbose=0)
        return np.argmax(action_probs[0])

    def train_experience_replay(self, batch_size):
        """Train on previous experiences in memory
        """
        mini_batch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []

        # DQN
        if self.strategy == "dqn":
            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate deep q-learning equation
                    next_state = next_state.reshape(1, 50, 14)
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                state = state.reshape(1, 50, 14)
                # estimate q-values based on current state
                q_values = self.model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        # DQN with fixed targets
        elif self.strategy == "t-dqn":
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    next_state = next_state.reshape(1, 50, 14)
                    # approximate deep q-learning equation with fixed targets
                    target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
                state = state.reshape(1, 50, 14)
                # estimate q-values based on current state
                q_values = self.model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        # Double DQN
        elif self.strategy == "double-dqn":
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    next_state = next_state.reshape(1, 50, 14)
                    # approximate double deep q-learning equation
                    target = reward + self.gamma * self.target_model.predict(
                        next_state)[0][np.argmax(self.model.predict(next_state)[0])]
                state = state.reshape(1, 50, 14)
                # estimate q-values based on current state
                q_values = self.model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        else:
            raise NotImplementedError()

        # update q-function parameters based on huber loss gradient
        loss = self.model.fit(
            np.array(X_train), np.array(y_train),
            epochs=1, verbose=0
        ).history["loss"][0]

        # as the training goes on we want the agent to
        # make less random and more optimal decisions
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, episode):
        self.model.save("models/{}__{}".format(self.model_name, episode))

    def load(self):
        return load_model("models/" + self.model_name, custom_objects=self.custom_objects)
