#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_belief(belief):
    
    plt.figure()
    
    ax = plt.subplot(2,1,1)
    ax.matshow(belief.reshape(1, belief.shape[0]))
    ax.set_xticks(np.arange(0, belief.shape[0],1))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks([])
    ax.title.set_text("Grid")
    
    ax = plt.subplot(2, 1, 2)
    ax.bar(np.arange(0, belief.shape[0]), belief)
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.set_ylim([0, 1.05])
    ax.title.set_text("Histogram")


def motion_model(action, belief):
    # add code here

    len_belief = len(belief)

    new_belief = np.zeros(len_belief)
    prob_correct = 0.75
    prob_no_move = 0.15
    prob_wrong = 0.10

    for i in range(len_belief):

        if action == "F":

            # correctly moved
            if i + 1 < len_belief:
                new_belief[i + 1] += belief[i] * prob_correct
            else:
                new_belief[i] += belief[i] * prob_correct

            # no move (robot stayed in the same place)
            new_belief[i] += belief[i] * prob_no_move

            # robot moved in the opposite direction
            if i - 1 >= 0:
                new_belief[i - 1] += belief[i] * prob_wrong
            else:
                new_belief[i] += belief[i] * prob_wrong

        elif action == "B":

            # correctly moved
            if i - 1 >= 0:
                new_belief[i - 1] += belief[i] * prob_correct
            else:
                new_belief[i] += belief[i] * prob_correct

            # no move (robot stayed in the same place)
            new_belief[i] += belief[i] * prob_no_move

            # robot moved in the opposite direction
            if i + 1 < len_belief:
                new_belief[i + 1] += belief[i] * prob_wrong
            else:
                new_belief[i] += belief[i] * prob_wrong
        
        else:
            raise ValueError("Invalid action. Use 'F' for forward and 'B' for backward.")
        
    return new_belief / np.sum(new_belief)

def sensor_model(observation, belief, world):
    # add code here
    pass 

def recursive_bayes_filter(actions, observations, belief, world):
    # add code here
    pass 

