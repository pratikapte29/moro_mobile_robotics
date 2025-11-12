#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import bresenham as bh

def plot_gridmap(gridmap):
    plt.figure()
    plt.imshow(gridmap, cmap='Greys',vmin=0, vmax=1)
    
def init_gridmap(size, res):
    gridmap = np.zeros([int(np.ceil(size/res)), int(np.ceil(size/res))])
    return gridmap

def world2map(pose, gridmap, map_res):
    origin = np.array(gridmap.shape)/2
    new_pose = np.zeros_like(pose)
    new_pose[0] = np.round(pose[0]/map_res) + origin[0];
    new_pose[1] = np.round(pose[1]/map_res) + origin[1];
    return new_pose.astype(int)

def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr    

def ranges2points(ranges):
    # laser properties
    start_angle = -1.5708
    angular_res = 0.0087270
    max_range = 30
    # rays within range
    num_beams = ranges.shape[0]
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points
    angles = np.linspace(start_angle, start_angle + (num_beams*angular_res), num_beams)[idx]
    points = np.array([np.multiply(ranges[idx], np.cos(angles)), np.multiply(ranges[idx], np.sin(angles))])
    # homogeneous points
    points_hom = np.append(points, np.ones((1, points.shape[1])), axis=0)
    return points_hom

def ranges2cells(r_ranges, w_pose, gridmap, map_res):
    # ranges to points
    r_points = ranges2points(r_ranges)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # covert to map frame
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0:2,:]
    return m_points

def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose  

def bresenham(x0, y0, x1, y1):
    l = np.array(list(bh.bresenham(x0, y0, x1, y1)))
    return l
    
def prob2logodds(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)  # to avoid division by zero or log of zero
    l = np.log(p / (1 - p))
    return l

def logodds2prob(l):
    l = np.clip(l, -50, 50)  # to avoid overflow in exp
    prob = 1 - (1 / (1 + np.exp(l)))
    return prob

"""
DOUBT:
for values where prob results to being indefinite, I have 
clipped it to the highest or near zero values. I am not 
sure if that is the right way.
"""


def inv_sensor_model(cell, endpoint, prob_occ, prob_free):

    # If the cell is the endpoint, it's occupied
    if np.array_equal(cell, endpoint):
        return prob_occ
    else:
        # All other cells along the ray are free
        return prob_free
    

def grid_mapping_with_known_poses(poses_raw, ranges_raw, map_res, occ_gridmap, prior, prob_free, prob_occ):

    # Convert prior probability to log-odds
    log_odds_prior = prob2logodds(prior)

    # Initialize gridmap with prior log-odds
    log_odds_grid = np.full_like(occ_gridmap, log_odds_prior, dtype=np.float64)

    # Update gridmap based on each pose and corresponding range measurements
    
    for r in range(ranges_raw.shape[0]):
        # convert single pose and single scan to map/frame cells
        pose_cell = poses2cells(poses_raw[r, :], occ_gridmap, map_res)
        endpoints = ranges2cells(ranges_raw[r, :], poses_raw[r, :], occ_gridmap, map_res)

        curr_pose = pose_cell

        for i in range(endpoints.shape[1]):
            endpoint = endpoints[:, i]

            # Get cells along the line from robot pose to endpoint
            line_cells = bresenham(curr_pose[0], curr_pose[1], endpoint[0], endpoint[1])

            # Update all of these cells along the line as "empty" i.e. free
            for j in range(len(line_cells) - 1):  
                cell = line_cells[j]
        
                if 0 <= cell[0] < occ_gridmap.shape[0] and 0 <= cell[1] < occ_gridmap.shape[1]:
                    prob_cell = inv_sensor_model(cell, endpoint, prob_occ, prob_free)
                    log_odds_update = prob2logodds(prob_cell)
                    log_odds_grid[cell[0], cell[1]] += log_odds_update - log_odds_prior 
    
            endpoint_cell = line_cells[-1]

            if 0 <= endpoint_cell[0] < occ_gridmap.shape[0] and 0 <= endpoint_cell[1] < occ_gridmap.shape[1]: 
                prob_cell = inv_sensor_model(endpoint_cell, endpoint, prob_occ, prob_free)
                log_odds_update = prob2logodds(prob_cell)
                log_odds_grid[endpoint_cell[0], endpoint_cell[1]] += log_odds_update - log_odds_prior 

    # Convert log-odds grid back to probability grid
    occ_gridmap = logodds2prob(log_odds_grid)

    return occ_gridmap
