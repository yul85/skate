import sys
import pydart2 as pydart
import numpy as np

from skate_cma.hpcma import HpCma
from skate_cma.PenaltyType import PenaltyType


def objective(i, penalty_option0, penalty_option1, penalty_weight):
    """
    Spiral
    """
    penalty_weight[PenaltyType.TORQUE] = 0.
    # penalty_weight[PenaltyType.PELVIS_HEADING] = 3.

    # penalty_weight[PenaltyType.RIGHT_FOOT_CONTACT] = 2.
    # penalty_weight[PenaltyType.RIGHT_FOOT_CONTACT_END] = 2.
    # penalty_weight[PenaltyType.LEFT_FOOT_CONTACT] = 4.
    # penalty_weight[PenaltyType.LEFT_FOOT_CONTACT_END] = 4.

    penalty_weight[PenaltyType.COM_IN_LEFT_FOOT] = 2.
    penalty_weight[PenaltyType.COM_IN_LEFT_FOOT_END] = 2.
    penalty_weight[PenaltyType.LEFT_FOOT_CONTACT] = 2.
    penalty_weight[PenaltyType.LEFT_FOOT_CONTACT_END] = 2.

    # penalty_option0[PenaltyType.COM_VEL] = np.array([0.5, 0.0, -0.0001])
    # penalty_option1[PenaltyType.COM_VEL] = np.array([0.5, 0.0, -0.0001])
    penalty_option0[PenaltyType.COM_VEL] = np.array([0.20769821*5.,  0., 0.00536799*5.])
    penalty_option1[PenaltyType.COM_VEL] = np.array([0.20769821*5.,  0., 0.00536799*5.])

    penalty_option0[PenaltyType.PELVIS_HEADING] = np.array([0.20769821, 0., 0.00536799])
    penalty_option1[PenaltyType.PELVIS_HEADING] = np.array([0.20769821, 0., 0.00536799])

    if i == 0:
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.88
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True
        # penalty_option0[PenaltyType.COM_VEL] = np.array([0.10976229, 0.0, 0.17668021])

        penalty_option1[PenaltyType.COM_HEIGHT] = 0.88
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_LEFT_FOOT_END] = True
        # penalty_option1[PenaltyType.COM_VEL] = np.array([0.10976229, 0.0, 0.17668021])

    elif i == 1:
        # backward turn
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.88
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT_END] = True
        # penalty_option0[PenaltyType.COM_VEL] = np.array([0.10976229, 0.0, 0.17668021])

if __name__ == '__main__':
    pydart.init()
    if len(sys.argv) == 1:
        cma = HpCma('spiral4_handmade', 24, sigma=0.1, max_time=3.0, cma_timeout=14400)
        # cma = HpCma('jump0507_2', 4, sigma=0.1, max_time=2., start_state_num=4, start_state_sol_dir='jump0507_2_model_201905192040/', cma_timeout=1)
    elif len(sys.argv) == 4:
        cma = HpCma(sys.argv[1], int(sys.argv[2]), sigma=float(sys.argv[3]), max_time=float(sys.argv[4]))
    elif len(sys.argv) == 8:
        cma = HpCma(sys.argv[1], int(sys.argv[2]), sigma=float(sys.argv[3]), max_time=float(sys.argv[4]), start_state_num=int(sys.argv[5]), start_state_sol_dir=sys.argv[6], cma_timeout=int(sys.argv[7]))
    else:
        cma = HpCma('spiral4_handmade', 1)

    dq = np.zeros(57)

    dq[3] = 0.20769821*5.
    dq[5] = 0.00536799*5.
    cma.set_init_dq(dq)
    cma.objective = objective
    cma.run()
