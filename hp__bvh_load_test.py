from fltk import Fl
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
from PyCommon.modules.Motion import ysMotion as ym
from PyCommon.modules.Motion import ysMotionLoader as yf
from dart_ik import DartIk
from PyCommon.modules.Math import mmMath as mm
import numpy as np

import pydart2 as pydart


def test(motion, ik, f):
    """

    :type motion: ym.JointMotion
    :type ik: DartIk
    :type f: int
    :return:
    """
    # left_elbow_index = motion[0].skeleton.getJointIndex('LeftElbow')
    # left_wrist_index = motion[0].skeleton.getJointIndex('LeftWrist')
    # left_knee_index = motion[0].skeleton.getJointIndex('LeftKnee')
    # left_ankle_index = motion[0].skeleton.getJointIndex('LeftAnkle')
    #
    # right_elbow_index = motion[0].skeleton.getJointIndex('RightElbow')
    # right_wrist_index = motion[0].skeleton.getJointIndex('RightWrist')
    # right_knee_index = motion[0].skeleton.getJointIndex('RightKnee')
    # right_ankle_index = motion[0].skeleton.getJointIndex('RightAnkle')
    #
    # head_index = motion[0].skeleton.getJointIndex('Head')
    # neck_index = motion[0].skeleton.getJointIndex('Neck')
    # chest_index = motion[0].skeleton.getJointIndex('Chest')
    # left_hip_index = motion[0].skeleton.getJointIndex('LeftHip')
    # right_hip_index = motion[0].skeleton.getJointIndex('RightHip')
    # left_toe_index = motion[0].skeleton.getJointIndex('LeftToe')
    # right_toe_index = motion[0].skeleton.getJointIndex('RightToe')
    #
    # q = ik.skel.q
    # q[3:6] = motion.getJointPositionGlobal(0, f)
    # ik.skel.set_positions(q)
    #
    # chest_pos = motion.getJointPositionGlobal(chest_index, f)
    # left_hip_vec = motion.getJointPositionGlobal(left_hip_index, f) - chest_pos
    # right_hip_vec = motion.getJointPositionGlobal(right_hip_index, f) - chest_pos
    #
    # hip_unit_x = np.asarray(mm.normalize(mm.cross(right_hip_vec, left_hip_vec)))
    # hip_unit_y = -np.asarray(mm.normalize(left_hip_vec+right_hip_vec))
    # hip_unit_z = np.asarray(mm.normalize(mm.cross(hip_unit_x, hip_unit_y)))
    #
    # hip_ori = mm.I_SO3()
    # hip_ori[:3, 0] = hip_unit_x
    # hip_ori[:3, 1] = hip_unit_y
    # hip_ori[:3, 2] = hip_unit_z
    #
    # left_blade_vec = motion.getJointPositionGlobal(left_toe_index, f) - motion.getJointPositionGlobal(left_ankle_index, f)
    # right_blade_vec = motion.getJointPositionGlobal(right_toe_index, f) - motion.getJointPositionGlobal(right_ankle_index, f)
    #
    # head_vec = motion.getJointPositionGlobal(head_index, f) - motion.getJointPositionGlobal(neck_index, f)
    #
    # ik.clean_constraints()
    # ik.add_orientation_const('h_pelvis', np.asarray(hip_ori))
    #
    # ik.add_joint_pos_const('j_abdomen', np.asarray(motion.getJointPositionGlobal(chest_index, f)))
    # ik.add_vector_const('h_head', mm.unitY(), np.asarray(head_vec))
    #
    # ik.add_joint_pos_const('j_forearm_left', np.asarray(motion.getJointPositionGlobal(left_elbow_index, f)))
    # ik.add_joint_pos_const('j_hand_left', np.asarray(motion.getJointPositionGlobal(left_wrist_index, f)))
    # ik.add_joint_pos_const('j_shin_left', np.asarray(motion.getJointPositionGlobal(left_knee_index, f)))
    # ik.add_joint_pos_const('j_heel_left', np.asarray(motion.getJointPositionGlobal(left_ankle_index, f)))
    #
    # ik.add_joint_pos_const('j_forearm_right', np.asarray(motion.getJointPositionGlobal(right_elbow_index, f)))
    # ik.add_joint_pos_const('j_hand_right', np.asarray(motion.getJointPositionGlobal(right_wrist_index, f)))
    # ik.add_joint_pos_const('j_shin_right', np.asarray(motion.getJointPositionGlobal(right_knee_index, f)))
    # ik.add_joint_pos_const('j_heel_right', np.asarray(motion.getJointPositionGlobal(right_ankle_index, f)))
    #
    # ik.add_vector_const('h_heel_left', np.array([1., -0.8, 0.]), np.asarray(left_blade_vec))
    # ik.add_vector_const('h_heel_right', np.array([1., -0.8, 0.]), np.asarray(right_blade_vec))
    #
    # ik.add_position_const('h_blade_left', motion.getJointPositionGlobal(left_toe_index, f), [-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0])
    # ik.add_position_const('h_blade_right', motion.getJointPositionGlobal(right_toe_index, f), [-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0])
    #
    # ik.solve()


def main():
    pydart.init()

    world = pydart.World(1./1000., 'data/skel/cart_pole_blade_3dof.skel')

    # viewer settings
    rd_contact_positions = [None]
    rd_contact_forces = [None]
    rd_target_position = [None]
    rd_com = [None]

    # motion = yf.readBvhFile('data/mocap/skate_spin.bvh', .0029, True)
    # motion = yf.readBvhFile('data/mocap/movingcam/backflip_a_result.bvh', .015, True)
    # motion = yf.readBvhFile('backflip_a.bvh', .01, True)
    motion = yf.readBvhFile('jump_result.bvh', .01, True)
    # motion = yf.readBvhFile('/home/yuri/Downloads/SfV_data/data/backflip_a.bvh', .01, True)
    # motion = yf.readBvhFile('/home/yuri/Downloads/SfV_data/data/jump.bvh', .01, True)
    # motion.translateByOffset([0., -0.5, 0.], True)
    ik = DartIk(world.skeletons[3])

    viewer = hsv.hpSimpleViewer(rect=[0, 0, 1280+300, 720+1+55], viewForceWnd=False)
    viewer.doc.addRenderer('MotionModel', yr.DartRenderer(world, (150,150,255), yr.POLYGON_FILL), visible=False)
    viewer.doc.addRenderer('motion', yr.JointMotionRenderer(motion))
    viewer.doc.addRenderer('target', yr.PointsRenderer(rd_target_position, (0, 255, 0)))
    viewer.doc.addRenderer('contact', yr.VectorsRenderer(rd_contact_forces, rd_contact_positions, (255,0,0)))
    viewer.doc.addRenderer('CM_plane', yr.PointsRenderer(rd_com, (0, 0, 255)))

    def preFrameCallback_Always(frame):
        motion.frame = frame

    def simulateCallback(frame):
        test(motion, ik, frame)

    viewer.setPreFrameCallback_Always(preFrameCallback_Always)
    viewer.setSimulateCallback(simulateCallback)
    viewer.setMaxFrame(len(motion)-1)
    viewer.startTimer(1./30.)
    viewer.show()

    Fl.run()


if __name__ == '__main__':
    main()
