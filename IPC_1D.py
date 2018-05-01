import numpy as np
import control as ctrl
import math

class Cart:
    def __init__(self, x, mass):
        self.x = x
        self.x_dot = 0.
        # self.y = int(0.6*world_size) 		# 0.6 was chosen for aesthetic reasons.
        self.mass = mass
        self.color = (0, 255, 0)


class Pendulum:
    def __init__(self, length, theta, ball_mass):
        self.length = length
        self.theta = theta
        self.theta_dot = 0.
        self.ball_mass = ball_mass
        self.color = (0,0,255)


def find_lqr_control_input(cart, pendulum, g):
    # Using LQR to find control inputs

    # The A and B matrices are derived in the report
    A = np.matrix([
        [0, 1, 0, 0],
        [0, 0, g * pendulum.ball_mass / cart.mass, 0],
        [0, 0, 0, 1],
        [0, 0, (cart.mass + pendulum.ball_mass) * g / (pendulum.length * cart.mass), 0]
    ])

    B = np.matrix([
        [0],
        [1 / cart.mass],
        [0],
        [1 / (pendulum.length * cart.mass)]
    ])

    # The Q and R matrices are emperically tuned. It is described further in the report
    Q = np.matrix([
        [1000, 0, 0, 0],
        [0, 0.01, 0, 0],
        [0, 0, 0.0000001, 0],
        [0, 0, 0, 100]
    ])

    R = np.matrix([10.])

    # The K matrix is calculated using the lqr function from the controls library
    K, S, E = ctrl.lqr(A, B, Q, R)
    np.matrix(K)

    x = np.matrix([
        [np.squeeze(np.asarray(cart.x))],
        [np.squeeze(np.asarray(cart.x_dot))],
        [np.squeeze(np.asarray(pendulum.theta))],
        [np.squeeze(np.asarray(pendulum.theta_dot))]
    ])

    desired = np.matrix([
        [2.],
        [0.05],
        [0.],
        [0]
    ])

    F = -np.dot(K, (x - desired))
    return np.squeeze(np.asarray(F))

def apply_control_input(cart, pendulum, F, time_delta, theta_dot, g):
    # Finding x and theta on considering the control inputs and the dynamics of the system
    theta_double_dot = (((cart.mass + pendulum.ball_mass) * g * math.sin(pendulum.theta)) + (F * math.cos(pendulum.theta)) - (pendulum.ball_mass * ((theta_dot)**2.0) * pendulum.length * math.sin(pendulum.theta) * math.cos(pendulum.theta))) / (pendulum.length * (cart.mass + (pendulum.ball_mass * (math.sin(pendulum.theta)**2.0))))
    x_double_dot = ((pendulum.ball_mass * g * math.sin(pendulum.theta) * math.cos(pendulum.theta)) - (pendulum.ball_mass * pendulum.length * math.sin(pendulum.theta) * (theta_dot**2)) + (F)) / (cart.mass + (pendulum.ball_mass * (math.sin(pendulum.theta)**2)))
    cart.x += cart.x_dot * time_delta
    pendulum.theta += pendulum.theta_dot * time_delta
    cart.x_dot += x_double_dot * time_delta
    pendulum.theta_dot = theta_double_dot * time_delta

    # cart.x += ((time_delta**2) * x_double_dot) + (((cart.x - x_tminus2) * time_delta) / previous_time_delta)
    # pendulum.theta += ((time_delta**2)*theta_double_dot) + (((pendulum.theta - theta_tminus2)*time_delta)/previous_time_delta)
