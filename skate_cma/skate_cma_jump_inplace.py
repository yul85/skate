import sys
import pydart2 as pydart
import numpy as np

from skate_cma.hpcma import HpCma
from skate_cma.PenaltyType import PenaltyType


def objective(i, penalty_option0, penalty_option1, penalty_weight):
    """
    0 : com h, left foot, #left foot com, max y velocity end, max y angvel end
    1 : am, axis, height com end, max y diff end
    2 : am, axis
    3 : right foot end, right foot end com
    4 : com height, right foot, right foot com, right foot com end, right foot end
    """

    penalty_weight[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = 0.1
    penalty_weight[PenaltyType.MAX_Y_VELOCITY_END] = 0.01
    penalty_weight[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_END] = 0.01
    penalty_weight[PenaltyType.MAX_Y_POS_DIFF_END] = 0.1
    penalty_weight[PenaltyType.TORQUE] = 0.

    if i == 0:
        # jump ready
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.64
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.MAX_Y_VELOCITY_END] = True
        penalty_option0[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_END] = True

        # jump max
        penalty_option1[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_HEAD] = True
        penalty_option1[PenaltyType.MAX_Y_POS_DIFF_END] = True
        penalty_option1[PenaltyType.COM_HEIGHT_END] = 1.45

    elif i == 1:
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

    elif i == 2:
        # jump down
        penalty_option0[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_HEAD] = True

        # prepare landing
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True

    elif i == 3:
        # prepare landing
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True

        # landed
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.75
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
    elif i == 4:
        # landed
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.75
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True


if __name__ == '__main__':
    pydart.init()
    if len(sys.argv) == 1:
        cma = HpCma('jump_0507_2', 4, sigma=0.1, max_time=2., cma_timeout=1)
        # cma = HpCma('jump0507_2', 4, sigma=0.1, max_time=2., start_state_num=4, start_state_sol_dir='jump0507_2_model_201905192040/', cma_timeout=1)
    elif len(sys.argv) == 5:
        cma = HpCma(sys.argv[1], int(sys.argv[2]), sigma=float(sys.argv[3]), max_time=float(sys.argv[4]))
    elif len(sys.argv) == 6:
        cma = HpCma(sys.argv[1], int(sys.argv[2]), sigma=float(sys.argv[3]), max_time=float(sys.argv[4]), cma_timeout=int(sys.argv[5]))
    elif len(sys.argv) == 8:
        cma = HpCma(sys.argv[1], int(sys.argv[2]), sigma=float(sys.argv[3]), max_time=float(sys.argv[4]), start_state_num=int(sys.argv[5]), start_state_sol_dir=sys.argv[6], cma_timeout=int(sys.argv[7]))
    else:
        cma = HpCma('jump_0507_2', 1)

    cma.objective = objective
    cma.run()
