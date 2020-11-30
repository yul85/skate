import sys
import pydart2 as pydart
import numpy as np

from skate_cma.hpcma2 import HpCma
from skate_cma.PenaltyType import PenaltyType


def objective(i, penalty_option0, penalty_option1, penalty_weight0, penalty_weight1):
    """
    Crossover NEW EXPERT VERSION!!!
    """

    penalty_weight0[PenaltyType.TORQUE] = 0.
    penalty_weight1[PenaltyType.TORQUE] = 0.

    if i < 10:
        # crossover
        penalty_weight0[PenaltyType.RIGHT_FOOT_CONTACT] = 2.
        penalty_weight0[PenaltyType.RIGHT_FOOT_CONTACT_END] = 2.

        penalty_weight0[PenaltyType.LEFT_FOOT_CONTACT] = 5.
        penalty_weight0[PenaltyType.LEFT_FOOT_CONTACT_END] = 5.

        penalty_weight1[PenaltyType.RIGHT_FOOT_CONTACT] = 2.
        penalty_weight1[PenaltyType.RIGHT_FOOT_CONTACT_END] = 2.

        penalty_weight1[PenaltyType.LEFT_FOOT_CONTACT] = 5.
        penalty_weight1[PenaltyType.LEFT_FOOT_CONTACT_END] = 5.

        penalty_option0[PenaltyType.COM_VEL] = np.array([1., 0.0, -0.001])
        penalty_option1[PenaltyType.COM_VEL] = np.array([1., 0.0, -0.001])
        penalty_option0[PenaltyType.PELVIS_HEADING] = np.array([1., 0.0, -0.001])
        penalty_option1[PenaltyType.PELVIS_HEADING] = np.array([1., 0.0, -0.001])
    elif i == 10:
        # transition from crossover to three turn

        penalty_weight0[PenaltyType.RIGHT_FOOT_CONTACT] = 2.
        penalty_weight0[PenaltyType.RIGHT_FOOT_CONTACT_END] = 2.

        penalty_weight0[PenaltyType.LEFT_FOOT_CONTACT] = 5.
        penalty_weight0[PenaltyType.LEFT_FOOT_CONTACT_END] = 5.

        penalty_weight1[PenaltyType.PELVIS_HEADING] = 3.
    elif i < 13:
        # threeturn
        penalty_weight0[PenaltyType.PELVIS_HEADING] = 3.
        penalty_weight1[PenaltyType.PELVIS_HEADING] = 3.
    elif i < 15:
        # three-turn to jump
        penalty_weight0[PenaltyType.PELVIS_HEADING] = 3.

        penalty_weight1[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = 0.1
        penalty_weight1[PenaltyType.MAX_Y_VELOCITY_END] = 0.01
        penalty_weight1[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_END] = 0.01
        penalty_weight1[PenaltyType.MAX_Y_POS_DIFF_END] = 0.1
        # penalty_weight0[PenaltyType.COM_IN_LEFT_FOOT] = 2.
        # penalty_weight1[PenaltyType.COM_IN_LEFT_FOOT] = 2.
        # penalty_weight0[PenaltyType.COM_IN_LEFT_FOOT_END] = 2.
        # penalty_weight1[PenaltyType.COM_IN_LEFT_FOOT_END] = 2.
    elif i < 19:
        # jump
        penalty_weight0[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = 0.1
        penalty_weight0[PenaltyType.MAX_Y_VELOCITY_END] = 0.01
        penalty_weight0[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_END] = 0.01
        penalty_weight0[PenaltyType.MAX_Y_POS_DIFF_END] = 0.1

        penalty_weight1[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = 0.1
        penalty_weight1[PenaltyType.MAX_Y_VELOCITY_END] = 0.01
        penalty_weight1[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_END] = 0.01
        penalty_weight1[PenaltyType.MAX_Y_POS_DIFF_END] = 0.1
    elif i == 19:
        # jump to three-turn
        penalty_weight0[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = 0.1
        penalty_weight0[PenaltyType.MAX_Y_VELOCITY_END] = 0.01
        penalty_weight0[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_END] = 0.01
        penalty_weight0[PenaltyType.MAX_Y_POS_DIFF_END] = 0.1

        penalty_weight1[PenaltyType.PELVIS_HEADING] = 3.
    elif i < 21:
        # threeturn back
        penalty_weight0[PenaltyType.PELVIS_HEADING] = 3.

        penalty_weight1[PenaltyType.PELVIS_HEADING] = 3.
    elif i == 21:
        # three-turn back to crossover
        penalty_weight0[PenaltyType.PELVIS_HEADING] = 3.

        penalty_weight1[PenaltyType.RIGHT_FOOT_CONTACT] = 2.
        penalty_weight1[PenaltyType.RIGHT_FOOT_CONTACT_END] = 2.

        penalty_weight1[PenaltyType.LEFT_FOOT_CONTACT] = 5.
        penalty_weight1[PenaltyType.LEFT_FOOT_CONTACT_END] = 5.
    else:
        # crossover
        penalty_weight0[PenaltyType.RIGHT_FOOT_CONTACT] = 2.
        penalty_weight0[PenaltyType.RIGHT_FOOT_CONTACT_END] = 2.

        penalty_weight0[PenaltyType.LEFT_FOOT_CONTACT] = 5.
        penalty_weight0[PenaltyType.LEFT_FOOT_CONTACT_END] = 5.

        penalty_weight1[PenaltyType.RIGHT_FOOT_CONTACT] = 2.
        penalty_weight1[PenaltyType.RIGHT_FOOT_CONTACT_END] = 2.

        penalty_weight1[PenaltyType.LEFT_FOOT_CONTACT] = 5.
        penalty_weight1[PenaltyType.LEFT_FOOT_CONTACT_END] = 5.

        penalty_option0[PenaltyType.COM_VEL] = np.array([1., 0.0, -0.001])
        penalty_option1[PenaltyType.COM_VEL] = np.array([1., 0.0, -0.001])
        penalty_option0[PenaltyType.PELVIS_HEADING] = np.array([1., 0.0, -0.001])
        penalty_option1[PenaltyType.PELVIS_HEADING] = np.array([1., 0.0, -0.001])

    if i == 0:
        # stand
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True

        # pump
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.83
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True

    elif i == 1:
        # pump
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.83
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True

        # X_leg
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_END] = True

    elif i == 2:
        # X_leg
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_END] = True

        # stand
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True

    elif i == 3:

        # stand
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True

        # pump
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.83
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True

    elif i == 4:
        # pump
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.83
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True

        # X_leg
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_END] = True

    elif i == 5:
        # X_leg
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_END] = True

        # stand
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True

    elif i == 6:
        # stand
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True

        # pump
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.83
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True

    elif i == 7:
        # pump
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.83
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True

        # X_leg
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_END] = True

    elif i == 8:
        # X_leg
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_END] = True

        # stand
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True

    elif i == 9:
        # stand
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True

        #transition state
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.89
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True

    elif i == 10:
        #transition state
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.89
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True

        # forward turn
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.89
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True
        penalty_option1[PenaltyType.COM_VEL] = np.array([1., 0.0, -0.001])
        penalty_option1[PenaltyType.PELVIS_HEADING] = np.array([1., 0.0, -0.001])

    elif i == 11:
        # forward turn
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.89
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True
        penalty_option0[PenaltyType.COM_VEL] = np.array([1., 0.0, -0.001])
        penalty_option0[PenaltyType.PELVIS_HEADING] = np.array([1., 0.0, -0.001])

        # backward turn
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.89
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True
        penalty_option1[PenaltyType.COM_VEL] = np.array([1., 0.0, -0.001])
        penalty_option1[PenaltyType.PELVIS_HEADING] = np.array([-1., 0.0, 0.001])

    elif i == 12:
        # backward turn
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.89
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True
        penalty_option0[PenaltyType.COM_VEL] = np.array([1., 0.0, -0.001])
        penalty_option0[PenaltyType.PELVIS_HEADING] = np.array([-1., 0.0, 0.001])

        #transition 1
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.8
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True
        penalty_option1[PenaltyType.COM_VEL] = np.array([1., 0.0, -0.001])
        penalty_option1[PenaltyType.PELVIS_HEADING] = np.array([-1., 0.0, 0.001])

    elif i == 13:
        # transition 1
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.8
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True
        penalty_option0[PenaltyType.COM_VEL] = np.array([1., 0.0, -0.001])
        penalty_option0[PenaltyType.PELVIS_HEADING] = np.array([-1., 0.0, 0.001])

        # transition 2
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.72
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True
        penalty_option1[PenaltyType.COM_VEL] = np.array([1., 0.0, -0.001])
        penalty_option1[PenaltyType.PELVIS_HEADING] = np.array([-1., 0.0, 0.001])

    elif i == 14:
        # transition 2
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.72
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True
        penalty_option0[PenaltyType.COM_VEL] = np.array([1., 0.0, -0.001])
        penalty_option0[PenaltyType.PELVIS_HEADING] = np.array([-1., 0.0, 0.001])

        # jump ready
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.64
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.MAX_Y_VELOCITY_END] = True
        penalty_option1[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_END] = True

    elif i == 15:
        # jump ready
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.64
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.MAX_Y_VELOCITY_END] = True
        penalty_option0[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_END] = True

        # jump max
        penalty_option1[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_HEAD] = True
        penalty_option1[PenaltyType.MAX_Y_POS_DIFF_END] = True
        penalty_option1[PenaltyType.COM_HEIGHT_END] = 1.45

    elif i == 16:
        # jump max
        penalty_option0[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_HEAD] = True
        penalty_option0[PenaltyType.MAX_Y_POS_DIFF_END] = True
        penalty_option0[PenaltyType.COM_HEIGHT_END] = 1.45

        # jump down
        penalty_option1[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_HEAD] = True

    elif i == 17:
        # jump down
        penalty_option0[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_HEAD] = True

        # prepare landing
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True

    elif i == 18:
        # prepare landing
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True

        # landed
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.75
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
    elif i == 19:
        # landed
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.75
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True

        # backward turn
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        penalty_option1[PenaltyType.COM_VEL] = np.array([-1., 0.0, 0.001])
        penalty_option1[PenaltyType.PELVIS_HEADING] = np.array([1., 0.0, .0])
    elif i == 20:
        # backward turn
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        penalty_option0[PenaltyType.COM_VEL] = np.array([-1., 0.0, 0.001])
        penalty_option0[PenaltyType.PELVIS_HEADING] = np.array([1., 0.0, .0])

        # middle
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.89
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        penalty_option1[PenaltyType.COM_VEL] = np.array([-1., 0.0, 0.001])
        penalty_option1[PenaltyType.PELVIS_HEADING] = np.array([0., 0.0, -1.0])

    elif i == 21:
        # middle
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.89
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        penalty_option0[PenaltyType.COM_VEL] = np.array([-1., 0.0, 0.001])
        penalty_option0[PenaltyType.PELVIS_HEADING] = np.array([0., 0.0, -1.0])

        # forward turn
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        penalty_option1[PenaltyType.COM_VEL] = np.array([-1., 0.0, 0.001])
        penalty_option1[PenaltyType.PELVIS_HEADING] = np.array([-1., 0.0, 0.0])

    elif i == 22:
        # forward turn
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        penalty_option0[PenaltyType.COM_VEL] = np.array([-1., 0.0, 0.001])
        penalty_option0[PenaltyType.PELVIS_HEADING] = np.array([-1., 0.0, 0.0])

        # stand
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True

    elif i == 23:
        # stand
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True

        # pump
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.83
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True

    elif i == 24:
        # pump
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.83
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True

        # X_leg
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_END] = True

    elif i == 25:
        # X_leg
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_END] = True

        # stand
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True

    elif i == 26:
        # stand
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True

        # pump
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.83
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True

    elif i == 27:
        # pump
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.83
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True

        # X_leg
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_END] = True

    elif i == 28:
        # X_leg
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_END] = True

        # stand
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True

    elif i == 29:
        # stand
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.87
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True


if __name__ == '__main__':
    pydart.init()
    if len(sys.argv) == 1:
        cma = HpCma('sequence4', 24, sigma=0.1, max_time=16.3, start_state_num=13, start_state_sol_dir='sequence3_model_201907101130/', cma_timeout=7200)
    elif len(sys.argv) == 4:
        cma = HpCma(sys.argv[1], int(sys.argv[2]), sigma=float(sys.argv[3]), max_time=float(sys.argv[4]))
    elif len(sys.argv) == 8:
        cma = HpCma(sys.argv[1], int(sys.argv[2]), sigma=float(sys.argv[3]), max_time=float(sys.argv[4]), start_state_num=int(sys.argv[5]), start_state_sol_dir=sys.argv[6], cma_timeout=int(sys.argv[7]))
    else:
        cma = HpCma('sequence4', 1)

    # dq = np.zeros(57)
    # dq[3] = 1.5
    # cma.set_init_dq(dq)
    cma.objective = objective
    cma.run()
