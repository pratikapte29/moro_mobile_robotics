#!/usr/bin/env python3

import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

FLAG_SENSOR_FAILURE = -1
FLAG_TIME_MISSMATCH = -2


def inverse_motion_model(pose, pose_prev):
    ##STUDENT_CODE: #TODO:Q1 compute rot1, trans, and rot2 of the inverse motion model






    ##END_STUDENT_CODE
    u = np.array([rot1, trans, rot2])
    return u


def ekf_predict(mu, S, u, R):
    ##STUDENT_CODE: #TODO:Q1 given the previous gaussian distribution (mu, S), the solution of the inverse motion model u, and the noise matrix R, update the gaussian and return the new (mu,S)



    # Update mu

    # Update covariance




    ##END_STUDENT_CODE
    return mu, S


def ekf_correct(mu, S, z, Q, M):
    # number of observations for the current measurements
    num_obs = z.shape[1]

    # Update for each observation
    for i in range(num_obs):
        ##STUDENT_CODE: #TODO:Q2 given the gaussian (mu,S), the observations z, the map M, and the observation noise matrix Q return the new gaussian (mu, S)



        # Obtain the ID of landmark for the current observation

        # Get map location of the landmark ID

        # Compute distance to observed landmark

        # Compute observation model H

        # Compute kalman gain K

        # Compute delta_mu

        # Update mu

        # Update covariance matrix S




    ##END_STUDENT_CODE
    return mu, S


def init_params():
    ##STUDENT_CODE: #TODO:Q3 Initialize belief (mu and S), process noise R, and measurement noise Q






    ##END_STUDENT_CODE

    return mu, S, R, Q


def run_ekf_localization(plot=True, step_size=100, last_step=1500, sensor_specs=None):
    dataset = pickle.load(open("dataset_2d_landmarks.p", "rb"))
    max_steps = len(dataset["odom"])

    odom_timesteps, sensor_timesteps = get_timesteps_system(
        sensor_specs, step_size, last_step, max_steps, plot=plot
    )

    mu, S, R, Q = init_params()

    # Read map
    M = dataset["M"]

    # visualize init
    axes = plot_state(M)

    for timestep_now, timestep_prev, sensor_timestep in zip(
        odom_timesteps[1:], odom_timesteps[:-1], sensor_timesteps[1:]
    ):
        ##STUDENT_CODE: #TODO:Q3 Implement predict and correct for loop!



        # Compute control command **ui** from odometry 

        ui = inverse_motion_model()

        # EKF prediction step

        # Compute EKF correction step if timestep_sensor is neither FLAG_TIME_MISSMATCH or FLAG_SENSOR_FAILURE!




        ##END_STUDENT_CODE

        # visualize result
        if plot and timestep_now % 10 == 0:
            color = (
                "green"
                if sensor_timestep != FLAG_SENSOR_FAILURE
                and sensor_timestep != FLAG_TIME_MISSMATCH
                else "blue"
            )
            plt.plot(mu[0], mu[1], "x", color=color)
            plot_2dcov_axes(axes, mu, S, color=color)

    if plot:
        gt = np.stack(dataset["gt"], axis=0)
        plt.plot(gt[:last_step, 0], gt[:last_step, 1], color="red")
        plt.show()
        print("_" * 100, "\n")

    else:
        return mu, S, ui


def get_timesteps_system(
    sensor_specs, odom_step_size, last_step, max_steps, plot=False
):
    if sensor_specs is None:
        sensor_timesteps = get_timesteps(odom_step_size, last_step, max_steps)
        odom_timesteps = get_timesteps(odom_step_size, last_step, max_steps)

    else:
        p_sensor_failure = sensor_specs["p_failure"]
        assert p_sensor_failure <= 1 or p_sensor_failure >= 0, (
            "Invalid p_failure percentage!"
        )

        sensor_step_size = int(sensor_specs["relative_frequency"] * odom_step_size)
        tolerance_steps = int(sensor_specs["relative_tolerance"] * odom_step_size)

        sensor_timesteps = get_timesteps(sensor_step_size, last_step, max_steps)
        odom_timesteps = get_timesteps(odom_step_size, last_step, max_steps)

        plot_timestep_A(sensor_timesteps, plot)

        ##STUDENT_CODE: #TODO:Q4 Task1: Sensor Failure - Sensor fails with p_sensor_failure -> Assign FLAG_SENSOR_FAILURE to the timestep






        ##END_STUDENT_CODE
        plot_timestep_B(sensor_timesteps, plot)

        ##STUDENT_CODE: #TODO:Q4 Task2: Frequency matching - Find the most recent sensor_timestep from sensor_timesteps



        #                    within tolerance_steps -> assign sensor_timestep

        #                           else            -> assign FLAG_TIME_MISSMATCH




        ##END_STUDENT_CODE

        plot_timestep_C(sensor_timesteps, odom_timesteps, tolerance_steps, plot)

    assert odom_timesteps.size == sensor_timesteps.size, (
        "sensor_timesteps should be of same length as odom_timesteps! Either a valid sensor timestep or None!"
    )

    if plot:
        plt.show()

    return odom_timesteps, sensor_timesteps


#################################################
#   Provided helper functions - Do not modify!  # 
#################################################

def plot_state(M):
    # initialize figure for 5.3
    plt.figure(figsize=(8, 6))
    # initialize figure
    axes = plt.gca()
    axes.set_xlim([-5, 25])
    axes.set_ylim([-5, 25])
    plt.title("EKF Localization")

    legend_elements = [
        Line2D([0], [0], marker="^", color="black", linestyle="", label="Map"),
        Line2D([0], [0], color="r", label="GroundTruth"),
        Line2D([0], [0], marker=".", color="b", linestyle="", label="Prediction Only"),
        Line2D(
            [0],
            [0],
            marker=".",
            color="green",
            linestyle="",
            label="Prediction & Correction",
        ),
    ]
    plt.legend(
        handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2
    )

    axes.set_xlim([np.min(M[:, 0]) - 2, np.max(M[:, 0]) + 2])
    axes.set_xlim([np.min(M[:, 1]) - 2, np.max(M[:, 1]) + 2])
    plt.plot(M[:, 0], M[:, 1], "^", color="black")
    plt.title("EKF Localization")

    return axes


def plot_2dcov(mu, cov):
    # covariance only in x,y
    d, v = np.linalg.eig(cov[:-1, :-1])

    # ellipse orientation
    a = np.sqrt(d[0])
    b = np.sqrt(d[1])

    # compute ellipse orientation
    if v[0, 0] == 0:
        theta = np.pi / 2
    else:
        theta = np.arctan2(v[0, 1], v[0, 0])

    # create an ellipse
    ellipse = Ellipse(
        (mu[0], mu[1]),
        width=a * 2,
        height=b * 2,
        angle=np.deg2rad(theta),
        edgecolor="blue",
        alpha=0.3,
    )

    ax = plt.gca()

    return ax.add_patch(ellipse)


def plot_2dcov_axes(axes, mu, cov, color):
    # covariance only in x,y
    d, v = np.linalg.eig(cov[:-1, :-1])

    # ellipse orientation
    a = np.sqrt(d[0])
    b = np.sqrt(d[1])

    # compute ellipse orientation
    if v[0, 0] == 0:
        theta = np.pi / 2
    else:
        theta = np.arctan2(v[0, 1], v[0, 0])

    # create an ellipse
    ellipse = Ellipse(
        (mu[0], mu[1]),
        width=a * 2,
        height=b * 2,
        angle=np.deg2rad(theta),
        edgecolor=color,
        facecolor="light" + color,
        alpha=0.3,
    )

    axes.add_patch(ellipse)


def plot_result_ekf(mode="predict", plot=True):
    assert mode in ["predict", "predict&correct"], f"Invalid mode {mode} provided!"

    dataset = pickle.load(open("dataset_2d_landmarks.p", "rb"))
    # Read map
    M = dataset["M"]

    # process noise: R
    sigma_x = 0.25  # [m]
    sigma_y = 0.25  # [m]
    sigma_theta = np.deg2rad(10)  # [rad]
    R = np.diag(np.array([sigma_x, sigma_y, sigma_theta]) ** 2)

    # 2x2 observation noise
    sigma_r = 0.3  # [m]
    sigma_phi = np.deg2rad(5)  # [rad]
    Q = np.diag(np.array([sigma_r, sigma_phi]) ** 2)

    # initialize state variables
    mu = dataset["gt"][0]
    S = np.zeros([3, 3])

    # initialize figure
    plt.figure(figsize=(8, 6))
    axes = plt.gca()
    plt.title("Prediction")
    axes.set_xlim(1.4, 2.5)

    # Controls
    u_1 = inverse_motion_model(dataset["odom"][50], dataset["odom"][0])
    u_2 = inverse_motion_model(dataset["odom"][100], dataset["odom"][50])
    u = [u_1, u_2]

    # Observations
    z_1 = dataset["z"][50]
    z_2 = dataset["z"][100]
    z = [z_1, z_2]

    # GroundTruth
    gt_1 = dataset["gt"][50]
    gt_2 = dataset["gt"][100]
    gt = [gt_1, gt_2]

    # Two Iterations of predicitions
    for u_i, gt_i, z_i, first_flag in zip(u, gt, z, [True, False]):
        if "predict" in mode:
            mu, S = ekf_predict(mu, S, u_i, R)

            color = "blue"
            label = "Prediction" if first_flag else "_"
            plt.plot(mu[0], mu[1], "x", color=color, label=label)
            plot_2dcov_axes(axes, mu, S, color=color)
            plt.draw()

        # Do additional correction step
        if "correct" in mode:
            mu, S = ekf_correct(mu, S, z_i, Q, M)

            color = "green"
            label = "Correction" if first_flag else "_"
            plt.plot(mu[0], mu[1], "x", color=color, label=label)
            plot_2dcov_axes(axes, mu, S, color=color)
            plt.draw()

        label = "GroundTruth" if first_flag else "_"
        plt.plot(gt_i[0], gt_i[1], ".r", label=label)

    # visualize result
    if plot:
        plt.legend()
        plt.show()
    else:
        return u_i, mu, S


def wrapToPi(theta):
    while theta < -np.pi:
        theta = theta + 2 * np.pi
    while theta > np.pi:
        theta = theta - 2 * np.pi
    return theta


def get_timesteps(step_size, last_step, total_steps):
    # Get timesteps to be plotted
    if last_step is None:
        last_step = total_steps
    else:
        last_step = min(total_steps, last_step)

    timesteps = list(range(0, last_step, step_size))

    return np.array(timesteps)


def plot_timestep_A(sensor_timesteps, plot):
    if plot:
        plt.figure(figsize=(7.4, 2))
        plt.scatter(
            sensor_timesteps,
            2 * np.ones_like(sensor_timesteps),
            marker="s",
            color="red",
            label="Failed Sensor",
        )


def plot_timestep_B(sensor_timesteps, plot):
    if plot:
        valid_mask = sensor_timesteps >= 0
        plt.scatter(
            sensor_timesteps[valid_mask],
            2 * np.ones_like(sensor_timesteps[valid_mask]),
            marker="s",
            color="orange",
            label="Sensor",
        )


def plot_timestep_C(sensor_timesteps, odom_timesteps, tolerance_steps, plot):
    if plot:
        valid_mask = sensor_timesteps >= 0

        plt.scatter(
            sensor_timesteps[valid_mask],
            2 * np.ones_like(sensor_timesteps[valid_mask]),
            marker="x",
            color="green",
            label="Matched Sensor",
        )
        plt.scatter(
            odom_timesteps,
            1 * np.ones_like(odom_timesteps),
            marker="x",
            color="b",
            label="Odometry",
        )

        for i, s in enumerate(sensor_timesteps):
            if s != FLAG_SENSOR_FAILURE and s != FLAG_TIME_MISSMATCH:
                plt.plot(
                    [odom_timesteps[i], s],
                    [1, 2],
                    color="gray",
                    linestyle="--",
                    alpha=0.5,
                )

        # Add labels
        plt.yticks([1, 2], ["Odometry", "Sensor"])
        plt.ylim([0.5, 2.5])
        plt.xlabel("Timestep")
        plt.title("Sensor vs Odometry Timesteps")
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=4)
        plt.xticks(odom_timesteps)
        plt.grid(True, axis="x", linestyle="--", alpha=0.5)

        for t in odom_timesteps:
            plt.axvspan(
                t - tolerance_steps, t, ymin=0.0, ymax=1.0, color="blue", alpha=0.1
            )
