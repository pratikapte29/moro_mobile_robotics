#!/usr/bin/env py_thon3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def inverse_motion_model(pose_t_1, pose_t):
    ##STUDENT_CODE:  #TODO Compute rot1,trans and rot2 of the inverse motion model.

    trans = ((pose_t_1[0] - pose_t[0])**2 + (pose_t_1[1] - pose_t[1])**2)**0.5

    rot1 = np.arctan2(pose_t_1[1] - pose_t[1], pose_t_1[0] - pose_t[0]) - pose_t[2]

    rot2 = pose_t_1[2] - pose_t[2] - rot1


    ##END_STUDENT_CODE:
    return rot1, trans, rot2


def probability_density(mean, variance):
    variance = max(variance,0.00001) #Avoid division by 0!

    ##STUDENT_CODE:  #TODO Compute normal distribution

    density = (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-(mean ** 2) / (2 * variance))

    ##END_STUDENT_CODE
    return density


def motion_model(x_t, x_t_1, u_t, alpha, marginalise_p3=False):

    
    ##STUDENT_CODE:  #TODO Compute p1, p2 and p3!

    # p1 = probability_density()

    # Extract odometry poses
    x_bar_t_1 = u_t[0] 
    x_bar_t = u_t[1] 

    # Use inverse motion model to calculate rot, trans 
    rot1, trans, rot2 = inverse_motion_model(x_bar_t_1, x_bar_t)
    rot1_cap, trans_cap, rot2_cap = inverse_motion_model(x_t_1, x_t)  # notation based on lecture slides

    # Calculate variances for eahc component
    var1 = alpha[0] * abs(rot1) + alpha[1] * trans
    var2 = alpha[2] * trans + alpha[3] * (abs(rot1) + abs(rot2))
    var3 = alpha[0] * abs(rot2) + alpha[1] * trans

    # Calculate probability densities
    p1 = probability_density(rot1 - rot1_cap, var1)
    p2 = probability_density(trans - trans_cap, var2)
    p3 = probability_density(rot2 - rot2_cap, var3)

    ##END_STUDENT_CODE 
    if marginalise_p3: 
        return p1 * p2
    else:
        return p1 * p2 * p3


def plot_posterior_belief(x_t_1, u_t, alpha, ret=False, marginalise_p3=False, N_theta = 8):
    size = 150
    gridmap = np.zeros([size, size])
    origin = [np.floor(size/2),np.floor(size/2)]
    res = 0.01

    
    if not marginalise_p3:
        marginalise_p3 = False
        dtheta = 2 * np.pi / N_theta
        

        ##STUDENT_CODE #TODO #3.4 A) Compute gridmap
        for i in range(size):
            for j in range(size):
                probability = 0

                for k in range(N_theta):
                    # Calculate pose for each cell and each theta
                    x_t = np.array([res * (i - origin[0]), 
                                    res * (j - origin[1]),
                                    k * dtheta])
                    
                    # Integrate over all possible thetas
                    prob = motion_model(x_t, x_t_1, u_t, alpha, marginalise_p3)
                    probability += prob * dtheta
                
                gridmap[i, j] = probability

        ##END_STUDENT_CODE 

    else:
        marginalise_p3 = True
        ##STUDENT_CODE #TODO #3.4 B) Use Marginalised motion_model marginalise_p3=True and pass to motion_model!
        dtheta = 2 * np.pi / N_theta
        
        for i in range(size):
            for j in range(size):
                probability = 0

                for k in range(N_theta):
                    # Calculate pose for each cell and each theta
                    x_t = np.array([res * (i - origin[0]), 
                                    res * (j - origin[1]),
                                    k * dtheta])
                    
                    # Integrate over all possible thetas
                    prob = motion_model(x_t, x_t_1, u_t, alpha, marginalise_p3)
                    probability += prob * dtheta
                
                gridmap[i, j] = probability


        ##END_STUDENT_CODE 

    if ret:
        return gridmap
    
    # Normalize
    assert np.sum(gridmap)!=0, "Sum over gridmap is zero! - Error or no implementation!"
    gridmap = gridmap/np.sum(gridmap)
    plt.imshow(1-gridmap, cmap="gray", extent=[-res*(size - origin[0]), res*(size - origin[0]), -res*(size - origin[1]), res*(size - origin[1])])
    plt.show()

def evaluate_sample_odometry(alpha, pose_0, odom, ret=False):
    plot_lims = [(0,6),(2.5,6.5)]

    # get ground truth poses
    gt_poses = [pose_0]
    last_pose = pose_0
    for odom_idx in range(len(odom) - 1):

        ##STUDENT_CODE #TODO: compute xt, yt, theta_t from odometry



        ##END_STUDENT_CODE
        current_pose = [x_t, y_t, theta_t]
        gt_poses.append(current_pose)
        last_pose = current_pose

    gt_poses = np.array(gt_poses)
   
    # draw samples incrementally
    current_pose = pose_0
    estimated_poses = [pose_0]
    samples_for_plot = []

    num_samples = 1000
    samples = np.zeros((num_samples, 3))
    samples[:, 0] = pose_0[0]
    samples[:, 1] = pose_0[1]

    for odom_idx in range(len(odom) - 1):
        # calculate the new samples

        ##STUDENT_CODE # TODO: update all samples and the current_pose



        ##END_STUDENT_CODE
        estimated_poses.append(current_pose)
        samples_for_plot.append(np.copy(samples))


    estimated_poses = np.array(estimated_poses)
    if ret:
        return gt_poses, estimated_poses

    print("estimated_poses: ", estimated_poses)

    #plot all samples, simga elipse and estimated and gt poses
    for idx, samples in enumerate(samples_for_plot):
        sc = plt.scatter(samples[:, 0], samples[:, 1],marker='.')
        color = sc.get_facecolor()[0]
        ax = plt.gca()
        confidence_ellipse(samples[:, 0], samples[:, 1], ax, edgecolor=color,n_std=2)
        plt.scatter(estimated_poses[idx+1, 0], estimated_poses[idx+1 ,1], marker='D',color=color,edgecolor='black')
        plt.scatter(gt_poses[idx+1, 0], gt_poses[idx+1, 1], marker='*',color=color,edgecolor='black', )

    plt.scatter(estimated_poses[0, 0], estimated_poses[0 ,1], marker='D',color='black',edgecolor='black',label="Estimated Pose")
    plt.scatter(gt_poses[0, 0], gt_poses[0, 1], marker='*',color='black',edgecolor='black',label="Ground Truth Pose")    
    plt.plot(0,0,label="2-Sigma Ellipse ~95.45%",color="black",lw=0.5)
        
    plt.xlim(plot_lims[0])
    plt.ylim(plot_lims[1])
    plt.legend()
    plt.show()

def plot_sample_motion_model(pose_t_1, u_t, alpha):
    num_samples = 1000
    samples = np.zeros((num_samples, 3))
    samples[:, 0] = pose_t_1[0]
    samples[:, 1] = pose_t_1[1]

    for i in range(num_samples):
        samples[i] = sample_motion_model(samples[i], u_t, alpha)

    plt.figure()
    plt.plot(samples[:,0], samples[:,1], '.')
    plt.axis('equal')
    plt.show()

def get_sample(std): 
    #Irwin–Hall -> Using Central Limit Theorem to generate cheap normal distributed samples. 
    tot = 0
    for i in range(12):
        tot += np.random.uniform(-std,std)
    
    return 0.5*tot


def sample_motion_model(pose_t_1, u_t, alpha):
    ##STUDENT_CODE #TODO Compute x_t, y_t and theta_t



    ##END_STUDENT_CODE
    return x_t, y_t, theta_t



from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
#https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)