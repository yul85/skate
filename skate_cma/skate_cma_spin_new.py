import sys
import pydart2 as pydart
import numpy as np

from skate_cma.hpcma import HpCma
from skate_cma.PenaltyType import PenaltyType


def objective(i, penalty_option0, penalty_option1, penalty_weight):
    """
    1 (double_stance): left foot, right foot, max y angvel end
    2 (spin ready1): right foot, right foot com, am, max y angvel end
    3 (spin ready2): right foot, right foot com, max_am_end
    4 (spin):  am, axis, right foot, right foot com,
    5 (end): right foot, right foot com, right foot com end, right foot end
    """
    penalty_weight[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = 1.
    penalty_weight[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_END] = 0.1

    penalty_weight[PenaltyType.LEFT_FOOT_CONTACT] = 5.
    penalty_weight[PenaltyType.LEFT_FOOT_CONTACT_END] = 5.
    penalty_weight[PenaltyType.RIGHT_FOOT_CONTACT] = 10.
    penalty_weight[PenaltyType.RIGHT_FOOT_CONTACT_END] = 10.
    penalty_weight[PenaltyType.LEFT_FOOT_TOE_CONTACT] = 5.
    penalty_weight[PenaltyType.LEFT_FOOT_TOE_CONTACT_END] = 5.
    penalty_weight[PenaltyType.RIGHT_FOOT_TOE_CONTACT] = 10.
    penalty_weight[PenaltyType.RIGHT_FOOT_TOE_CONTACT_END] = 10.
    penalty_weight[PenaltyType.COM_IN_RIGHT_FOOT_TOE] = 5.
    penalty_weight[PenaltyType.COM_IN_RIGHT_FOOT_TOE_END] = 5.
    penalty_weight[PenaltyType.TORQUE] = 0.
    penalty_weight[PenaltyType.MAX_Y_RIGHT_FOOT_HEEL_END] = 0.1

    if i == 0:
        # double_stance
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.LEFT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_IN_LEFT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        # penalty_option0[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_END] = True
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.74

        # spin ready1
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        # penalty_option1[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = True
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.88

    elif i == 1:
        # spin ready1
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        # penalty_option0[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = True
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.88

        # spin ready2
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        # penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_TOE] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_TOE_END] = True
        # penalty_option1[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = True
        # penalty_option1[PenaltyType.COM_HEIGHT] = 0.89
        # penalty_option1[PenaltyType.COM_HEIGHT_END] = 0.93
        penalty_option1[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_END] = True
        penalty_option1[PenaltyType.MAX_Y_RIGHT_FOOT_HEEL] = True
        penalty_option1[PenaltyType.MAX_Y_RIGHT_FOOT_HEEL_END] = True

    elif i == 2:
        # spin ready2
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        # penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_TOE] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_TOE_END] = True
        # penalty_option0[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = True
        # penalty_option0[PenaltyType.COM_HEIGHT] = 0.89
        # penalty_option0[PenaltyType.COM_HEIGHT_END] = 0.93
        penalty_option0[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_END] = True
        penalty_option0[PenaltyType.MAX_Y_RIGHT_FOOT_HEEL] = True
        penalty_option0[PenaltyType.MAX_Y_RIGHT_FOOT_HEEL_END] = True

        # spin
        penalty_option1[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = True
        penalty_option1[PenaltyType.MAX_Y_RIGHT_FOOT_HEEL] = True
        penalty_option1[PenaltyType.COM_IN_HEAD] = True
        # penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT] = True
        # penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        # penalty_option1[PenaltyType.COM_HEIGHT] = 0.93
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_TOE] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_TOE_END] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_TOE_CONTACT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_TOE_CONTACT_END] = True

    elif i == 3:
        # spin
        penalty_option0[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] = True
        penalty_option0[PenaltyType.MAX_Y_RIGHT_FOOT_HEEL] = True
        penalty_option0[PenaltyType.COM_IN_HEAD] = True
        # penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT] = True
        # penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        # penalty_option0[PenaltyType.COM_HEIGHT] = 0.93
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_TOE] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_TOE_END] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_TOE_CONTACT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_TOE_CONTACT_END] = True

        # end
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option1[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        penalty_option1[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option1[PenaltyType.COM_HEIGHT] = 0.72

    elif i == 4:
        # end
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT] = True
        penalty_option0[PenaltyType.COM_IN_RIGHT_FOOT_END] = True
        penalty_option0[PenaltyType.RIGHT_FOOT_CONTACT_END] = True
        penalty_option0[PenaltyType.COM_HEIGHT] = 0.72


if __name__ == '__main__':
    pydart.init()
    if len(sys.argv) == 1:
        cma = HpCma('hmr_skating_spin_new1', 24, sigma=0.1, max_time=2.3, cma_timeout=1800)
    elif len(sys.argv) == 5:
        cma = HpCma(sys.argv[1], int(sys.argv[2]), sigma=float(sys.argv[3]), max_time=float(sys.argv[4]))
    elif len(sys.argv) == 6:
        cma = HpCma(sys.argv[1], int(sys.argv[2]), sigma=float(sys.argv[3]), max_time=float(sys.argv[4]), cma_timeout=int(sys.argv[5]))
    elif len(sys.argv) == 8:
        """
        sys.argv[1] : env_name
        sys.argv[2] : # of cores
        sys.argv[3] : 0.1
        sys.argv[4] : max time
        sys.argv[5] : starting state num
        sys.argv[6] : if sys.argv[5] is not 0, directory to import
        sys.argv[7] : cma timeout(1800 default)
        """
        cma = HpCma(sys.argv[1], int(sys.argv[2]), sigma=float(sys.argv[3]), max_time=float(sys.argv[4]), start_state_num=int(sys.argv[5]), start_state_sol_dir=sys.argv[6], cma_timeout=int(sys.argv[7]))
    else:
        cma = HpCma('hmr_skating_spin_new1', 1)
        # cma = HpCma('jump0507_2', 4, sigma=0.1, max_time=2., start_state_num=1, start_state_sol_dir='jump0507_2_model_201905191614/')

    dq = np.zeros(57)
    dq[3] = 1.
    cma.set_init_dq(dq)
    cma.objective = objective
    cma.run()
