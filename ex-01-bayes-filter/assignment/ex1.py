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
    len_belief = len(belief)
    new_belief = np.zeros(len_belief)
    
    prob_correct = 0.75
    prob_no_move = 0.15
    prob_wrong = 0.10

    for i in range(len_belief):

        if action == "F":
            # Correctly moved forward
            if i + 1 < len_belief:
                new_belief[i + 1] += belief[i] * prob_correct
            else:
                # At right border: can't move forward, so stay
                new_belief[i] += belief[i] * prob_correct

            # No move (robot stayed in the same place)
            new_belief[i] += belief[i] * prob_no_move

            # Robot moved backward (opposite direction)
            if i - 1 >= 0:
                new_belief[i - 1] += belief[i] * prob_wrong
            else:
                # At left border: can't move backward, so stay
                new_belief[i] += belief[i] * prob_wrong

        elif action == "B":
            # Correctly moved backward
            if i - 1 >= 0:
                new_belief[i - 1] += belief[i] * prob_correct
            else:
                # At left border: can't move backward, so stay
                new_belief[i] += belief[i] * prob_correct

            # No move (robot stayed in the same place)
            new_belief[i] += belief[i] * prob_no_move

            # Robot moved forward (opposite direction)
            if i + 1 < len_belief:
                new_belief[i + 1] += belief[i] * prob_wrong
            else:
                # At right border: can't move forward, so stay
                new_belief[i] += belief[i] * prob_wrong
        
        else:
            raise ValueError("Invalid action. Use 'F' for forward and 'B' for backward.")
    
    return new_belief 


def sensor_model(observation, belief, world):
    len_belief = len(belief)
    new_belief = np.zeros(len_belief)
    
    # Sensor probabilities
    prob_white_correct = 0.75  
    prob_blue_correct = 0.85   
    
    for i in range(len_belief):
        # Get actual color at position i
        actual_color = world[i]
        
        # Calculate likelihood: P(observation | at position i)
        if actual_color == observation:
            # Colors match - sensor detected correctly
            if actual_color == 1:  # white
                likelihood = prob_white_correct  # 0.75
            else:  # blue (0)
                likelihood = prob_blue_correct   # 0.85
        else:
            # Colors don't match - sensor made an error
            if actual_color == 1:  # actual white, but observed blue
                likelihood = 1 - prob_white_correct  # 0.25
            else:  # actual blue, but observed white
                likelihood = 1 - prob_blue_correct   # 0.15
        
        # THIS IS WHERE YOU MULTIPLY!
        new_belief[i] = likelihood * belief[i]
    
    # Normalize so probabilities sum to 1
    return new_belief / np.sum(new_belief)


def recursive_bayes_filter(actions, observations, belief, world):
    
    for action, observation in zip(actions, observations):
        # Step 1: PREDICTION - apply motion model
        belief = motion_model(action, belief)
        
        # Step 2: CORRECTION - apply sensor model
        belief = sensor_model(observation, belief, world)
    
    return belief

