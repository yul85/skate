import sys
import pydart2 as pydart
import numpy as np

from skate_cma.hpcma import HpCma
from skate_cma.PenaltyType import PenaltyType


def objective(i, penalty_option0, penalty_option1, penalty_weight):
    """
    Crossover
    """

    penalty_weight[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = 0.1
    penalty_weight[PenaltyType.MAX_Y_VELOCITY_END] = 0.01
    penalty_weight[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_END] = 0.01
    penalty_weight[PenaltyType.MAX_Y_POS_DIFF_END] = 0.1

    penalty_option0[PenaltyType.COM_VEL] = np.array([0.5, 0.0, -0.0001])
    penalty_option1[PenaltyType.COM_VEL] = np.array([0.5, 0.0, -0.0001])

    if i == 0:
        # stand
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.88
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True

        # pump
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.78
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True

    elif i == 1:
        # pump
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.78
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True

        # stand
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.88
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True

    elif i == 2:
        # stand
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.88
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True

        # lift right leg
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.9
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True

    elif i == 3:
        # lift right leg
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.9
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True

        # X_leg
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.88
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True

    elif i == 4:
        # X_leg
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.88
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True

        # stand
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.88
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True

    elif i == 5:
        # stand
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.88
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True

        # pump
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.78
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True

    elif i == 6:
        # pump
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.78
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True

        # stand
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.88
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True

    elif i == 7:
        # stand
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.88
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True

        # lift right leg
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.9
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True

    elif i == 8:
        # lift right leg
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.9
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True

        # X_leg
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.88
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True

    elif i == 9:
        # X_leg
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.88
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True


if __name__ == '__main__':
    pydart.init()
    if len(sys.argv) == 1:
        cma = HpCma('hmr_skating_crossover_iterate', 24, sigma=0.1, max_time=2., cma_timeout=3600)
        # cma = HpCma('jump0507_2', 4, sigma=0.1, max_time=2., start_state_num=4, start_state_sol_dir='jump0507_2_model_201905192040/', cma_timeout=1)
    elif len(sys.argv) == 4:
        cma = HpCma(sys.argv[1], int(sys.argv[2]), sigma=float(sys.argv[3]), max_time=float(sys.argv[4]))
    elif len(sys.argv) == 8:
        cma = HpCma(sys.argv[1], int(sys.argv[2]), sigma=float(sys.argv[3]), max_time=float(sys.argv[4]), start_state_num=int(sys.argv[5]), start_state_sol_dir=sys.argv[6], cma_timeout=int(sys.argv[7]))
    else:
        cma = HpCma('hmr_skating_crossover_iterate', 1)

    # dq = np.zeros(57)
    # dq[3] = 1.5
    # cma.set_init_dq(dq)
    cma.objective = objective
    cma.run()
