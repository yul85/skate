import sys
import pydart2 as pydart
import numpy as np

from skate_cma.hpcma import HpCma
from skate_cma.PenaltyType import PenaltyType


def objective(i, penalty_option0, penalty_option1):
    penalty_option0[PenaltyType.COM_HEIGHT] = 0.6
    penalty_option1[PenaltyType.COM_HEIGHT] = 0.6
    if i == 0:
        # drop
        penalty_option0[PenaltyType.COM_HEIGHT] = None
        penalty_option0[PenaltyType.COM_HEIGHT_END] = 0.6
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True

        # lr
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
    elif i == 1:
        # lr
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True

        # l
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.COM_VEL_END] = np.array([0.5, 0., -0.2])
    elif i == 2:
        # l
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.COM_VEL_END] = np.array([0.5, 0., -0.2])

        # l
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.COM_VEL_END] = np.array([0.5, 0., -0.2])
    elif i == 3:
        # l
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.COM_VEL_END] = np.array([0.5, 0., -0.2])

        # lr
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
    elif i == 4:
        # lr
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True

        # r
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.COM_VEL_END] = np.array([0.5, 0., 0.2])
    elif i == 5:
        # r
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.COM_VEL_END] = np.array([0.5, 0., 0.2])

        # lr
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
    elif i == 6:
        # lr
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True

        # l
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.COM_VEL_END] = np.array([0.5, 0., -0.2])
    elif i == 7:
        # l
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.COM_VEL_END] = np.array([0.5, 0., -0.2])

        # l
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.COM_VEL_END] = np.array([0.5, 0., -0.2])
    elif i == 8:
        # l
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.COM_VEL_END] = np.array([0.5, 0., -0.2])

        # lr
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True


if __name__ == '__main__':
    pydart.init()
    if len(sys.argv) == 1:
        cma = HpCma('hmr_skating_basic4', 4, sigma=0.05, max_time=4.)
    else:
        cma = HpCma(sys.argv[1], int(sys.argv[2]), sigma=0.05, max_time=float(sys.argv[3]))
    cma.objective = objective
    cma.run()
