import numpy as np
import pydart2 as pydart
import math
import dart_ik
import time

from fltk import *
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
from PyCommon.modules.Simulator import yulQpSimulator_penalty as yulqp
# from PyCommon.modules.Simulator import yulQpSimulator_inequality_blade as yulqp
# from PyCommon.modules.Simulator import yulQpSimulator as yulqp
# from PyCommon.modules.Simulator import yulQpSimulator_noknifecon as yulqp
from PyCommon.modules.Motion import ysMotionLoader as yf
from PyCommon.modules.Math import mmMath as mm

render_vector = []
render_vector_origin = []
push_force = []
push_force_origin = []
push_force1 = []
push_force1_origin = []
blade_force = []
blade_force_origin = []

COM = []

l_blade_dir_vec = []
l_blade_dir_vec_origin = []
r_blade_dir_vec = []
r_blade_dir_vec_origin = []

ref_l_blade_dir_vec = []
ref_l_blade_dir_vec_origin = []
ref_r_blade_dir_vec = []
ref_r_blade_dir_vec_origin = []

r1_point = []
r2_point = []

new_bvh_load = False
new_bvh_load = True

def TicTocGenerator():
    ti = 0              # initial time
    tf = time.time()    # final time
    while True:
        ti = tf
        ti = time.time()
        yield ti - tf    # returns the time difference

TicToc = TicTocGenerator()      #create an instance of the TicToc generator

def toc(tempBool = True):
    #Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds.\n" %tempTimeInterval)

def tic():
    #Records a time in TicToc, marks the beginning
    toc(False)

class MyWorld(pydart.World):
    def __init__(self, ):
        pydart.World.__init__(self, 1.0 / 1000.0, './data/skel/blade2_3dof.skel')

        self.force = None
        self.force1 = None
        self.my_tau = None
        self.my_tau2 = None
        self.my_force = None
        self.my_force2 = None

        # ---------------------------------- bvh -----------------------------------
        bvh = yf.readBvhFileAsBvh('./data/mocap/skate_spin.bvh')
        motion = yf.readBvhFile('./data/mocap/skate_spin.bvh', 0.0029, True)

        self.bvh = bvh
        self.motion = motion
        self.duration = bvh.frameTime
        fn = bvh.frameNum
        self.fn = fn

        if new_bvh_load:
            tic()
            self.ik = dart_ik.DartIk(self.skeletons[1])

            self.q_list = []
            # hips_index = motion[0].skeleton.getJointIndex('Hips')

            left_elbow_index = motion[0].skeleton.getJointIndex('LeftElbow')
            left_wrist_index = motion[0].skeleton.getJointIndex('LeftWrist')
            left_knee_index = motion[0].skeleton.getJointIndex('LeftKnee')
            left_ankle_index = motion[0].skeleton.getJointIndex('LeftAnkle')

            right_elbow_index = motion[0].skeleton.getJointIndex('RightElbow')
            right_wrist_index = motion[0].skeleton.getJointIndex('RightWrist')
            right_knee_index = motion[0].skeleton.getJointIndex('RightKnee')
            right_ankle_index = motion[0].skeleton.getJointIndex('RightAnkle')

            head_index = motion[0].skeleton.getJointIndex('Head')
            neck_index = motion[0].skeleton.getJointIndex('Neck')
            chest_index = motion[0].skeleton.getJointIndex('Chest')
            left_hip_index = motion[0].skeleton.getJointIndex('LeftHip')
            right_hip_index = motion[0].skeleton.getJointIndex('RightHip')
            left_toe_index = motion[0].skeleton.getJointIndex('LeftToe')
            right_toe_index = motion[0].skeleton.getJointIndex('RightToe')

            left_shoulder_index = motion[0].skeleton.getJointIndex('LeftShoulder')
            right_shoulder_index = motion[0].skeleton.getJointIndex('RightShoulder')


            for f in range(fn):
                # motion.getJointPositionGlobal(left_knee_index)
                q = self.skeletons[1].q
                q[3:6] = motion.getJointPositionGlobal(0, f)
                # q[3:6] = motion[f].local_ts[0]
                self.skeletons[1].set_positions(q)

                chest_pos = motion.getJointPositionGlobal(chest_index, f)
                left_hip_vec = motion.getJointPositionGlobal(left_hip_index, f) - chest_pos
                right_hip_vec = motion.getJointPositionGlobal(right_hip_index, f) - chest_pos

                hip_unit_x = np.asarray(mm.normalize(mm.cross(right_hip_vec, left_hip_vec)))
                hip_unit_y = -np.asarray(mm.normalize(left_hip_vec + right_hip_vec))
                hip_unit_z = np.asarray(mm.normalize(mm.cross(hip_unit_x, hip_unit_y)))

                hip_ori = mm.I_SO3()
                hip_ori[:3, 0] = hip_unit_x
                hip_ori[:3, 1] = hip_unit_y
                hip_ori[:3, 2] = hip_unit_z

                left_blade_vec = motion.getJointPositionGlobal(left_toe_index, f) - motion.getJointPositionGlobal(
                    left_ankle_index, f)
                right_blade_vec = motion.getJointPositionGlobal(right_toe_index, f) - motion.getJointPositionGlobal(
                    right_ankle_index, f)

                head_vec = motion.getJointPositionGlobal(head_index, f) - motion.getJointPositionGlobal(neck_index, f)

                self.ik.clean_constraints()
                self.ik.add_orientation_const('h_pelvis', np.asarray(hip_ori))

                # self.ik.add_joint_pos_const('j_abdomen', np.asarray(motion.getJointPositionGlobal(chest_index, f)))
                self.ik.add_vector_const('h_head', mm.unitY(), np.asarray(head_vec))

                self.ik.add_joint_pos_const('j_scapula_left', np.asarray(motion.getJointPositionGlobal(left_shoulder_index, f)))
                self.ik.add_joint_pos_const('j_forearm_left', np.asarray(motion.getJointPositionGlobal(left_elbow_index, f)))
                self.ik.add_joint_pos_const('j_hand_left', np.asarray(motion.getJointPositionGlobal(left_wrist_index, f)))
                self.ik.add_joint_pos_const('j_shin_left', np.asarray(motion.getJointPositionGlobal(left_knee_index, f)))
                self.ik.add_joint_pos_const('j_heel_left', np.asarray(motion.getJointPositionGlobal(left_ankle_index, f)))

                self.ik.add_joint_pos_const('j_scapula_right', np.asarray(motion.getJointPositionGlobal(right_shoulder_index, f)))

                self.ik.add_joint_pos_const('j_forearm_right', np.asarray(motion.getJointPositionGlobal(right_elbow_index, f)))
                self.ik.add_joint_pos_const('j_hand_right', np.asarray(motion.getJointPositionGlobal(right_wrist_index, f)))
                self.ik.add_joint_pos_const('j_shin_right', np.asarray(motion.getJointPositionGlobal(right_knee_index, f)))
                self.ik.add_joint_pos_const('j_heel_right', np.asarray(motion.getJointPositionGlobal(right_ankle_index, f)))

                self.ik.add_vector_const('h_heel_left', np.array([1., -0.8, 0.]), np.asarray(left_blade_vec))
                self.ik.add_vector_const('h_heel_right', np.array([1., -0.8, 0.]), np.asarray(right_blade_vec))

                left_toe_pos = motion.getJointPositionGlobal(left_toe_index, f)
                # print("lt: ", left_toe_pos)
                # left_toe_pos[1] = 0.0
                self.ik.add_position_const('h_blade_left', left_toe_pos, [-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0])
                self.ik.add_position_const('h_blade_left', left_toe_pos, [0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0])
                right_toe_pos = motion.getJointPositionGlobal(right_toe_index, f)
                # right_toe_pos[1] = 0.0
                self.ik.add_position_const('h_blade_right', right_toe_pos, [-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0])
                self.ik.add_position_const('h_blade_right', right_toe_pos, [0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0])

                self.ik.solve()

                self.q_list.append(self.skeletons[1].q)

            #store q_list to txt file
            # with open('data/mocap/skate_spin.txt', 'w') as f:
            with open('data/mocap/skate_spin_test2.txt', 'w') as f:
                for pose in self.q_list:
                    for a in pose:
                        f.write('%s ' % a)
                    f.write('\n')
            # print(len(self.q_list[0]))
            # print(self.q_list)
            toc()

        # read q_list from txt file
        self.read_q_list = []
        # with open('data/mocap/skate_spin.txt') as f:
        with open('data/mocap/skate_spin_test2.txt') as f:
            # lines = f.read().splitlines()
            for line in f:
                q_temp = []
                line = line.replace(' \n', '')
                values = line.split(" ")
                for a in values:
                    q_temp.append(float(a))
                self.read_q_list.append(q_temp)
        #
        # print(self.read_q_list)
        # print(len(self.read_q_list))
        #---------------------------------- bvh -----------------------------------

        skel = self.skeletons[0]
        # print("mass: ", skel.m, "kg")

        # print('[Joint]')
        # for joint in skel.joints:
        #     print("\t" + str(joint))
        #     print("\t\tparent = " + str(joint.parent_bodynode))
        #     print("\t\tchild = " + str(joint.child_bodynode))
        #     print("\t\tdofs = " + str(joint.dofs))
        #
        # skel.joint("j_abdomen").set_position_upper_limit(10, 0.0)

        self.skeletons[1].set_positions(self.read_q_list[0])

        self.elapsedTime = 0
        self.fn_bvh = 0
        self.flag = 0

        self.l_dir = np.array([0., 0., 0.])
        self.r_dir = np.array([0., 0., 0.])
        self.l_blade_dir = np.array([0., 0., 0.])
        self.r_blade_dir = np.array([0., 0., 0.])
        self.r1 = np.array([0., 0., 0.])
        self.r2 = np.array([0., 0., 0.])

        self.contact_force = []
        self.contactPositionLocals = []
        self.bodyIDs = []
        # print("dof: ", skel.ndofs)

    def step(self):
        # if self.fn_bvh == 0 and self.flag < 200:
        #     print("here? : ", self.fn_bvh, self.flag)
        #     self.skeletons[1].set_positions(self.read_q_list[self.fn_bvh])
        #     self.flag += 1
        # else:
        #     if self.fn_bvh < 10:
        #         self.force = np.array([0.0, 0.0, 50.0])
        #     else:
        #         self.force = None
        #
        #     if self.flag < 10:
        #         self.skeletons[1].set_positions(self.read_q_list[self.fn_bvh])
        #         self.flag += 1
        #     else:
        #         self.flag = 0
        #         self.fn_bvh += 1

        # if self.fn_bvh < 10:
        #     self.force = np.array([0.0, 0.0, 50.0])
        # else:
        #     self.force = None

        ref_skel = self.skeletons[1]

        # q_temp = np.asarray(ref_skel.q)
        # cur_com_pos = ref_skel.com()
        #
        # ref_skel.set_positions(self.read_q_list[self.fn_bvh + 1])
        # future_com_pos = ref_skel.com()
        # ref_skel.set_positions(q_temp)
        #
        # ref_com_vel = future_com_pos - cur_com_pos
        # # print("COM_VEL", ref_com_vel)
        # self.force = 1000.* ref_com_vel

        if self.flag < 10:
            self.skeletons[1].set_positions(self.read_q_list[self.fn_bvh])
            self.flag += 1
        else:
            self.flag = 0
            self.fn_bvh += 1

        ndofs = skel.num_dofs()
        h = self.time_step()

        # PD CONTROL

        q_revised = np.asarray(self.read_q_list[self.fn_bvh])
        q_revised[0:1] = 0.001

        # gain_value = 25.0
        gain_value = 50.0
        Kp = np.diagflat([0.0] * 6 + [gain_value] * (ndofs - 6))
        Kd = np.diagflat([0.0] * 6 + [2. * (gain_value ** .5)] * (ndofs - 6))
        invM = np.linalg.inv(skel.M + Kd * h)
        p = -Kp.dot(skel.q - q_revised + skel.dq * h)
        d = -Kd.dot(skel.dq)
        qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        des_accel = p + d + qddot

        ddc = np.zeros(6)

        if self.fn_bvh < self.fn:
            q_temp = np.asarray(ref_skel.q)
            cur_blade_left_pos = ref_skel.body('h_blade_left').to_world([0., 0., 0.])
            cur_blade_right_pos = ref_skel.body('h_blade_right').to_world([0., 0., 0.])

            cur_v1 = ref_skel.body('h_blade_left').to_world([-0.1040 + 0.0216, 0.0, 0.0])
            cur_v2 = ref_skel.body('h_blade_left').to_world([0.1040 + 0.0216, 0.0, 0.0])
            blade_direction_vec_l = cur_v2 - cur_v1
            cur_blade_direction_vec_l = np.array([1, 0, 1]) * blade_direction_vec_l

            cur_v1 = ref_skel.body('h_blade_right').to_world([-0.1040 + 0.0216, 0.0, 0.0])
            cur_v2 = ref_skel.body('h_blade_right').to_world([0.1040 + 0.0216, 0.0, 0.0])
            blade_direction_vec_r = cur_v2 - cur_v1
            cur_blade_direction_vec_r = np.array([1, 0, 1]) * blade_direction_vec_r

            if np.linalg.norm(cur_blade_direction_vec_l) != 0:
                cur_blade_direction_vec_l = cur_blade_direction_vec_l / np.linalg.norm(cur_blade_direction_vec_l)
            if np.linalg.norm(cur_blade_direction_vec_r) != 0:
                cur_blade_direction_vec_r = cur_blade_direction_vec_r / np.linalg.norm(cur_blade_direction_vec_r)

            ref_skel.set_positions(self.read_q_list[self.fn_bvh+1])
            future_blade_left_pos = ref_skel.body('h_blade_left').to_world([0., 0., 0.])
            future_blade_right_pos = ref_skel.body('h_blade_right').to_world([0., 0., 0.])

            slidingAxisLeft = future_blade_left_pos - cur_blade_left_pos
            slidingAxisRight = future_blade_right_pos - cur_blade_right_pos
            slidingAxisLeft = np.array([1, 0, 1]) * slidingAxisLeft
            slidingAxisRight = np.array([1, 0, 1]) * slidingAxisRight

            if np.linalg.norm(slidingAxisLeft) != 0:
                slidingAxisLeft = slidingAxisLeft / np.linalg.norm(slidingAxisLeft)
            if np.linalg.norm(slidingAxisRight) != 0:
                slidingAxisRight = slidingAxisRight / np.linalg.norm(slidingAxisRight)

            r_vec_l = skel.body('h_blade_left').to_local([0., 0., 0.])
            r_vec_l = np.array([1, 0, 1]) * r_vec_l
            r_vec_r = skel.body('h_blade_right').to_local([0., 0., 0.])
            r_vec_r = np.array([1, 0, 1]) * r_vec_r
            r_vec_ref_l = ref_skel.body('h_blade_left').to_local([0., 0., 0.])
            r_vec_ref_l = np.array([1, 0, 1]) * r_vec_ref_l
            r_vec_ref_r = ref_skel.body('h_blade_right').to_local([0., 0., 0.])
            r_vec_ref_r = np.array([1, 0, 1]) * r_vec_ref_r
            zero_term = np.linalg.norm(r_vec_l - r_vec_ref_l) + np.linalg.norm(r_vec_r - r_vec_ref_r)
            ddc[0:3] = 1/(2*1.5)*zero_term + 0.1*(skel.com_velocity())

            ref_skel.set_positions(q_temp)

        self.l_dir = slidingAxisLeft
        self.r_dir = slidingAxisRight
        self.l_blade_dir = cur_blade_direction_vec_l
        self.r_blade_dir = cur_blade_direction_vec_r

        # YUL QP solve
        _ddq, _tau, _bodyIDs, _contactPositions, _contactPositionLocals, _contactForces = yulqp.calc_QP(
            skel, des_accel, ddc, slidingAxisLeft, slidingAxisRight, cur_blade_direction_vec_l, cur_blade_direction_vec_r, 1. / self.time_step())

        del self.contact_force[:]
        del self.bodyIDs[:]
        del self.contactPositionLocals[:]

        self.bodyIDs = _bodyIDs

        for i in range(len(_bodyIDs)):
            skel.body(_bodyIDs[i]).add_ext_force(_contactForces[i], _contactPositionLocals[i])
            self.contact_force.append(_contactForces[i])
            self.contactPositionLocals.append(_contactPositionLocals[i])
        # dartModel.applyPenaltyForce(_bodyIDs, _contactPositionLocals, _contactForces)

        # External force
        # if self.fn_bvh < 10:
        #     self.force = 15.*np.array([0.0, -1.0, 100.0])
        #     # self.force1 = 1.*np.array([-20.0, 0.0, -100.0])
        #
        #     # self.force = 1. * np.array([-1.0, -1.0, 100.0])
        #     # self.force1 = 1. * np.array([.0, -0.0, -100.0])
        # else:
        #     self.force = None
        #     self.force1 = None

        if self.force is not None:
            self.skeletons[0].body('h_pelvis').add_ext_force(self.force, [0., 0., -0.1088])
        if self.force1 is not None:
            self.skeletons[0].body('h_pelvis').add_ext_force(self.force1, [0., 0., 0.1088])

        # Jacobian transpose control
        if self.fn_bvh < 20:
            my_jaco = skel.body("h_blade_right").linear_jacobian()
            my_jaco_t = my_jaco.transpose()
            self.my_force2 = 30. * np.array([-5.0,  0., -5.0])
            # self.my_force2 = 10. * slidingAxisRight
            self.my_tau2 = np.dot(my_jaco_t, self.my_force2)

            p1 = skel.body("h_blade_left").to_world([-0.1040 + 0.0216, 0.0, 0.0])
            p2 = skel.body("h_blade_left").to_world([0.1040 + 0.0216, 0.0, 0.0])

            blade_direction_vec = p2 - p1
            blade_direction_vec[1] = -0.
            if np.linalg.norm(blade_direction_vec) != 0:
                blade_direction_vec = blade_direction_vec / np.linalg.norm(blade_direction_vec)

            # print("dir vec:", blade_direction_vec)
            my_jaco = skel.body("h_blade_left").linear_jacobian()
            my_jaco_t = my_jaco.transpose()
            self.my_force = 20. * np.array([2., 0., 10.])
            # self.my_force = 30. * np.array([1., -15., 40.])
            # self.my_force = 50. * blade_direction_vec
            # self.my_force = 500. * slidingAxisLeft
            self.my_tau = np.dot(my_jaco_t, self.my_force)

            _tau += self.my_tau + self.my_tau2
        else:
            self.my_tau = None
            self.my_tau2 = None

        # Jacobian transpose control
        # if self.fn_bvh < 10:
        #     my_jaco = skel.body("h_blade_right").linear_jacobian()
        #     my_jaco_t = my_jaco.transpose()
        #     self.my_force2 = 30. * np.array([-12.0,  -5., -10.0])
        #     # self.my_force2 = 10. * slidingAxisRight
        #     self.my_tau2 = np.dot(my_jaco_t, self.my_force2)
        #
        #     p1 = skel.body("h_blade_left").to_world([-0.1040 + 0.0216, 0.0, 0.0])
        #     p2 = skel.body("h_blade_left").to_world([0.1040 + 0.0216, 0.0, 0.0])
        #
        #     blade_direction_vec = p2 - p1
        #     blade_direction_vec[1] = -0.
        #     # print("dir vec:", blade_direction_vec)
        #     my_jaco = skel.body("h_blade_left").linear_jacobian()
        #     my_jaco_t = my_jaco.transpose()
        #     self.my_force = 50. * np.array([5., -5., 20.])
        #     # self.my_force = 30. * np.array([1., -15., 40.])
        #     # self.my_force = 100. * blade_direction_vec
        #     # self.my_force = 500. * slidingAxisLeft
        #     self.my_tau = np.dot(my_jaco_t, self.my_force)
        #
        #     _tau += self.my_tau + self.my_tau2
        # # else:
        # #     self.my_tau = None
        # #     self.my_tau2 = None
        #
        #
        # if self.fn_bvh < 20 and self.fn_bvh >= 10:
        #     my_jaco = skel.body("h_blade_right").linear_jacobian()
        #     my_jaco_t = my_jaco.transpose()
        #     self.my_force2 = 30. * np.array([5.,  -5., -10.0])
        #     # self.my_force2 = 10. * slidingAxisRight
        #     self.my_tau2 = np.dot(my_jaco_t, self.my_force2)
        #
        #     p1 = skel.body("h_blade_left").to_world([-0.1040 + 0.0216, 0.0, 0.0])
        #     p2 = skel.body("h_blade_left").to_world([0.1040 + 0.0216, 0.0, 0.0])
        #
        #     blade_direction_vec = p2 - p1
        #     blade_direction_vec[1] = -0.
        #     # print("dir vec:", blade_direction_vec)
        #     my_jaco = skel.body("h_blade_left").linear_jacobian()
        #     my_jaco_t = my_jaco.transpose()
        #     self.my_force = 50. * np.array([2., -1., 10.])
        #     # self.my_force = 30. * np.array([1., -15., 40.])
        #     # self.my_force = 100. * blade_direction_vec
        #     # self.my_force = 500. * slidingAxisLeft
        #     self.my_tau = np.dot(my_jaco_t, self.my_force)
        #
        #     _tau += self.my_tau + self.my_tau2
        # # else:
        # #     self.my_tau = None
        # #     self.my_tau2 = None
        #
        # if self.fn_bvh >= 20:
        #     my_jaco = skel.body("h_blade_right").linear_jacobian()
        #     my_jaco_t = my_jaco.transpose()
        #     self.my_force2 = 30. * np.array([0., -5., -0.0])
        #     self.my_force2 = 10. * slidingAxisRight
        #     self.my_tau2 = np.dot(my_jaco_t, self.my_force2)
        #
        #     my_jaco = skel.body("h_blade_left").linear_jacobian()
        #     my_jaco_t = my_jaco.transpose()
        #     self.my_force = 50. * np.array([0., -5., 0.])
        #     self.my_force = 10. * slidingAxisLeft
        #     self.my_tau = np.dot(my_jaco_t, self.my_force)
        #
        #     _tau += self.my_tau + self.my_tau2
        #
        #     # self.my_tau = None
        #     # self.my_tau2 = None

        skel.set_forces(_tau)

        super(MyWorld, self).step()

        # skel.set_positions(q)

    def on_key_press(self, key):
        if key == '1':
            self.force = np.array([100.0, 0.0, 0.0])
            self.duration = 1000
            print('push backward: f = %s' % self.force)
        elif key == '2':
            self.force = np.array([-100.0, 0.0, 0.0])
            self.duration = 100
            print('push backward: f = %s' % self.force)

    def render_with_ys(self):
        # render contact force --yul
        contact_force = self.contact_force
        del render_vector[:]
        del render_vector_origin[:]
        del push_force[:]
        del push_force_origin[:]
        del push_force1[:]
        del push_force1_origin[:]
        del blade_force[:]
        del blade_force_origin[:]
        del COM[:]

        del l_blade_dir_vec[:]
        del l_blade_dir_vec_origin[:]
        del r_blade_dir_vec[:]
        del r_blade_dir_vec_origin[:]

        del ref_l_blade_dir_vec[:]
        del ref_l_blade_dir_vec_origin[:]
        del ref_r_blade_dir_vec[:]
        del ref_r_blade_dir_vec_origin[:]

        # del r1_point[:]
        # del r2_point[:]
        #
        # r1_point.append(self.r1)
        # r2_point.append(self.r2)

        com = self.skeletons[0].C
        com[1] = -0.99 + 0.05

        l_blade_dir_vec.append(self.l_dir)
        l_blade_dir_vec_origin.append(self.skeletons[1].body('h_blade_left').to_world())

        r_blade_dir_vec.append(self.r_dir)
        r_blade_dir_vec_origin.append(self.skeletons[1].body('h_blade_right').to_world())

        ref_l_blade_dir_vec.append(self.l_blade_dir)
        ref_l_blade_dir_vec_origin.append(self.skeletons[1].body('h_blade_left').to_world())

        ref_r_blade_dir_vec.append(self.r_blade_dir)
        ref_r_blade_dir_vec_origin.append(self.skeletons[1].body('h_blade_right').to_world())

        # com = self.skeletons[0].body('h_blade_left').to_world(np.array([0.1040 + 0.0216, +0.80354016 - 0.85354016, -0.054]))
        COM.append(com)
        if self.my_tau is not None:
            blade_force.append(self.my_force* 0.05)
            blade_force_origin.append(self.skeletons[0].body('h_heel_left').to_world())
        if self.my_tau2 is not None:
            blade_force.append(self.my_force2* 0.05)
            blade_force_origin.append(self.skeletons[0].body('h_heel_right').to_world())

        if self.force is not None:
            push_force.append(self.force * 0.001)
            push_force_origin.append(self.skeletons[0].body('h_pelvis').to_world([0., 0., -0.1088]))
        if self.force1 is not None:
            push_force1.append(self.force1 * 0.001)
            push_force1_origin.append(self.skeletons[0].body('h_pelvis').to_world([0., 0., 0.1088]))
        if len(self.bodyIDs) != 0:
            print(len(self.bodyIDs))
            for ii in range(len(self.contact_force)):
                if self.bodyIDs[ii] == 4:
                    body = self.skeletons[0].body('h_blade_left')
                else:
                    body = self.skeletons[0].body('h_blade_right')

                render_vector.append(contact_force[ii] / 100.)
                render_vector_origin.append(body.to_world(self.contactPositionLocals[ii]))


if __name__ == '__main__':
    print('Example: Skating -- Spin')

    pydart.init()
    print('pydart initialization OK')

    world = MyWorld()
    print('MyWorld  OK')
    ground = pydart.World(1. / 1000., './data/skel/ground.skel')

    skel = world.skeletons[0]
    q = np.asarray(skel.q)
    skel.set_positions(world.read_q_list[0])
    q_new = np.asarray(skel.q)
    # q_new[0:1] = 0.001
    # q_new[3:6] = q[3:6] - np.array([0., 0.025, 0.])
    # q_new[3:6] = q[3:6] - np.array([0., 0.035, 0.])
    q_new[3:6] = q[3:6] - np.array([0., 0.05, 0.])
    # q_new[17:18] = -0.2
    # q_new[20:21] = -0.2
    skel.set_positions(q_new)

    q_vel = skel.position_differences(world.read_q_list[0], world.read_q_list[1])/world.bvh.frameTime
    # print("Init vel: ", q_vel)
    skel.set_velocities(q_vel)

    # pydart.gui.viewer.launch_pyqt5(world)
    viewer = hsv.hpSimpleViewer(viewForceWnd=False)
    # viewer = hsv.hpSimpleViewer(rect=[0, 0, 1280 + 300, 720 + 1 + 55], viewForceWnd=False)

    viewer.doc.addRenderer('controlModel', yr.DartRenderer(world, (255, 255, 255), yr.POLYGON_FILL))
    viewer.doc.addRenderer('ground', yr.DartRenderer(ground, (255, 255, 255), yr.POLYGON_FILL))
    # viewer.doc.addRenderer('ground', yr.DartRenderer(ground, (255, 255, 255), yr.POLYGON_FILL), visible=False)
    viewer.doc.addRenderer('contactForce', yr.VectorsRenderer(render_vector, render_vector_origin, (255, 0, 0)))
    viewer.doc.addRenderer('pushForce', yr.WideArrowRenderer(push_force, push_force_origin, (0, 255, 0)))
    viewer.doc.addRenderer('pushForce1', yr.WideArrowRenderer(push_force1, push_force1_origin, (0, 255, 0)))
    viewer.doc.addRenderer('COM_pos', yr.PointsRenderer(COM))
    # viewer.doc.addRenderer('ref_motion', yr.JointMotionRenderer(world.motion), visible=False)

    viewer.motionViewWnd.glWindow.pOnPlaneshadow = (0., -0.98 + 0.0251, 0.)

    viewer.doc.addRenderer('bladeForce', yr.WideArrowRenderer(blade_force, blade_force_origin, (0, 0, 255)))

    viewer.doc.addRenderer('lbladeDir', yr.WideArrowRenderer(l_blade_dir_vec, l_blade_dir_vec_origin, (0, 255, 0)), visible=False)
    viewer.doc.addRenderer('rbladeDir', yr.WideArrowRenderer(r_blade_dir_vec, r_blade_dir_vec_origin, (0, 0, 255)), visible=False)
    viewer.doc.addRenderer('ref_lbladeDir', yr.WideArrowRenderer(ref_l_blade_dir_vec, ref_l_blade_dir_vec_origin, (0, 150, 0)), visible=False)
    viewer.doc.addRenderer('ref_rbladeDir', yr.WideArrowRenderer(ref_r_blade_dir_vec, ref_r_blade_dir_vec_origin, (0, 0, 150)), visible=False)

    # viewer.doc.addRenderer('r1', yr.PointsRenderer(r1_point))
    # viewer.doc.addRenderer('r2', yr.PointsRenderer(r2_point))

    viewer.motionViewWnd.glWindow.planeHeight = -0.98 + 0.0251

    def simulateCallback(frame):
        world.motion.frame = frame
        for i in range(10):
            world.step()
        world.render_with_ys()

    viewer.setSimulateCallback(simulateCallback)
    # viewer.setMaxFrame(1000)
    viewer.setMaxFrame(len(world.motion) - 1)
    # viewer.startTimer(1 / 25.)
    viewer.startTimer(1. / 30.)
    viewer.show()

    Fl.run()
