import sys
import pydart2 as pydart
import numpy as np

from skate_cma.hpcma import HpCma
from skate_cma.PenaltyType import PenaltyType


def objective(i, penalty_option0, penalty_option1, penalty_weight):
    penalty_weight[PenaltyType.TORQUE] = 0.

    penalty_option0[PenaltyType.COM_HEIGHT] = 0.6
    penalty_option1[PenaltyType.COM_HEIGHT] = 0.6

    penalty_option0[PenaltyType.COM_VEL_DIR_WITH_PELVIS_LOCAL] = np.array([0.5, 0., -0.2])
    penalty_option1[PenaltyType.COM_VEL_DIR_WITH_PELVIS_LOCAL] = np.array([0.5, 0., -0.2])
    penalty_weight[PenaltyType.COM_VEL_DIR_WITH_PELVIS_LOCAL] = 0.1

    penalty_weight[PenaltyType.RIGHT_FOOT_CONTACT] = 3.
    penalty_weight[PenaltyType.RIGHT_FOOT_CONTACT_END] = 3.
    penalty_weight[PenaltyType.LEFT_FOOT_CONTACT] = 3.
    penalty_weight[PenaltyType.LEFT_FOOT_CONTACT_END] = 3.

    if i == 0:
        # drop
        penalty_option0[PenaltyType.COM_HEIGHT] = None
        penalty_option0[PenaltyType.COM_HEIGHT_END] = 0.6
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True

        # lr
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_RIGHT_CENTER] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
    elif i % 4 == 1:
        # lr
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_RIGHT_CENTER] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True

        # l
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
    elif i % 4 == 2:
        # l
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True

        # lr
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_RIGHT_CENTER] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
    elif i % 4 == 3:
        # lr
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_RIGHT_CENTER] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True

        # r
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
    elif i % 4 == 0:
        # r
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True

        # lr
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_RIGHT_CENTER] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True


if __name__ == '__main__':
    pydart.init()
    if len(sys.argv) == 1:
        cma = HpCma('hmr_skating_basic5', 24, sigma=0.25, max_time=4., cma_timeout=7200)
    else:
        cma = HpCma(sys.argv[1], int(sys.argv[2]), sigma=0.1, max_time=float(sys.argv[3]))
    cma.objective = objective
    cma.run()
