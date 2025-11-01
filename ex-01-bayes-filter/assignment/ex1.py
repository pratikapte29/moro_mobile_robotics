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
    n = len(belief)
    new_belief = np.zeros(n)
    
    prob_correct = 0.75
    prob_stay = 0.15
    prob_opposite = 0.10
    
    for i in range(n):
        if action == 'B':  # Backward command
            if i == 0:  # Left boundary - can't go further left
                new_belief[i] += (prob_correct + prob_stay) * belief[i]  # Stay (can't go left + normal stay)
                new_belief[i + 1] += prob_opposite * belief[i]  # Go right (opposite)
            elif i == n - 1:  # Right boundary
                new_belief[i - 1] += prob_correct * belief[i]  # Go left (correct)
                new_belief[i] += (prob_opposite + prob_stay) * belief[i]  # Stay (can't go right + normal stay)
            else:  # Normal case
                new_belief[i - 1] += prob_correct * belief[i]  # Go left (correct)
                new_belief[i] += prob_stay * belief[i]         # Stay
                new_belief[i + 1] += prob_opposite * belief[i] # Go right (opposite)

        elif action == 'F':  # Forward command
            if i == 0:  # Left boundary
                new_belief[i + 1] += prob_correct * belief[i]  # Go right (correct)
                new_belief[i] += (prob_opposite + prob_stay) * belief[i]  # Stay (can't go left + normal stay)
            elif i == n - 1:  # Right boundary - can't go further right
                new_belief[i] += (prob_correct + prob_stay) * belief[i]  # Stay (can't go right + normal stay)
                new_belief[i - 1] += prob_opposite * belief[i]  # Go left (opposite)
            else:  # Normal case
                new_belief[i + 1] += prob_correct * belief[i]  # Go right (correct)
                new_belief[i] += prob_stay * belief[i]         # Stay
                new_belief[i - 1] += prob_opposite * belief[i] # Go left (opposite)
    
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

