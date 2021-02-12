from State import AI_Board
import os
import random
import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import Adam

import cv2

def build_network(num_actions):

    print("Initializing model ....")
    model = Sequential()
    model.add(Conv2D(32, (8, 8), padding='same',
                     strides=(4, 4), input_shape=(80, 160, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), padding='same', strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), padding='same', strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(num_actions))

    if os.path.exists("dqn.h5"):
        print("Loading weights from dqn.h5 .....")
        model.load_weights("dqn.h5")
        print("Weights loaded successfully.")
    adam = Adam(lr=1e-4)
    model.compile(loss='mse', optimizer=adam)
    print("Finished building model.")

    return model


def process(input):
    # resize image to 80x80 from 288x404
    image = cv2.resize(input, (160, 80))
    # scale down pixels values to (0,1)
    image = image / 255.0
    return image


def train_network():
    game = AI_Board()
    model = build_network(game.action_num)
    num_actions = game.action_num  # number of valid actions
    discount = 0.99  # decay rate of past observations
    observe = 200  # timesteps to observe before training
    explore = 3000000  # frames over which to anneal epsilon
    FINAL_EPSILON = 0.0001  # final value of epsilon
    INITIAL_EPSILON = 0.1  # starting value of epsilon
    replay_memory = 300  # number of previous transitions to remember

    epsilon = INITIAL_EPSILON
    timestep = 0
    loss = 0
    # initialize an instance of game
    # store the previous observations in replay memory
    replay = deque()
    image, _, reward, alive = game.next(0)
    # preprocess the image and stack to 80x80x4 pixels
    input_image = process(image)
    input_image = input_image.reshape(
        1, input_image.shape[0], input_image.shape[1], input_image.shape[2])

    while (True):
        if random.random() <= epsilon:
            action = random.randint(0, num_actions)
        else:
            q = model.predict(input_image)
            action = np.argmax(q)
        # decay epsilon linearly
        if epsilon > FINAL_EPSILON and timestep > observe:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / explore
        image1, _, reward, alive = game.next(action)
        image1 = process(image1)
        input_image1 = image1.reshape(1, image1.shape[0], image1.shape[1], image1.shape[2])

        replay.append((input_image, action, reward, input_image1, alive))
        if len(replay) > replay_memory:
            replay.popleft()

        if timestep > observe:
            try:
                # sample a minibatch of size 32 from replay memory
                minibatch = random.sample(replay, 16)
                s, a, r, s1, alive = zip(*minibatch)
                s = np.concatenate(s)
                s1 = np.concatenate(s1)
                targets = model.predict(s)
                print(s.shape, s1.shape, targets.shape)
                targets[range(16), a] = r + discount * \
                    np.max(model.predict(s1), axis=1)*alive
                loss += model.train_on_batch(s, targets)
            except Exception as e:
                print(e)
                continue

        input_image = input_image1
        timestep = timestep + 1

        if timestep % 400 == 0:
            model.save_weights("dqn.h5", overwrite=True)
        print("TIMESTEP: " + str(timestep) + ", EPSILON: " + str(epsilon) +
              ", ACTION: " + str(action) + ", REWARD: " + str(reward) + ", Loss: " + str(loss))
        loss = 0


if __name__ == "__main__":
    
    train_network()
    