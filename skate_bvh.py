import numpy as np
import pydart2 as pydart
import math
import dart_ik
import time

from fltk import *
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
from PyCommon.modules.Simulator import hpDartQpSimulator as hqp
from PyCommon.modules.Motion import ysMotionLoader as yf
from PyCommon.modules.Math import mmMath as mm

render_vector = []
render_vector_origin = []
push_force = []
push_force_origin = []
blade_force = []
blade_force_origin = []

rd_footCenter = []

new_bvh_load = False
# new_bvh_load = True

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

        # ---------------------------------- bvh -----------------------------------
        bvh = yf.readBvhFileAsBvh('./data/mocap/skate_spin.bvh')
        motion = yf.readBvhFile('./data/mocap/skate_spin.bvh', 0.0029, True)
        self.motion = motion
        self.duration = bvh.frameTime
        fn = bvh.frameNum

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

                self.ik.add_position_const('h_blade_left', motion.getJointPositionGlobal(left_toe_index, f),
                                      [-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0])
                # self.ik.add_position_const('h_blade_left', motion.getJointPositionGlobal(left_toe_index, f),
                #                            [0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0])
                self.ik.add_position_const('h_blade_right', motion.getJointPositionGlobal(right_toe_index, f),
                                      [-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0])
                # self.ik.add_position_const('h_blade_right', motion.getJointPositionGlobal(right_toe_index, f),
                #                            [0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0])

                self.ik.solve()

                self.q_list.append(self.skeletons[1].q)

            #store q_list to txt file
            with open('data/mocap/skate_spin.txt', 'w') as f:
                for pose in self.q_list:
                    for a in pose:
                        f.write('%s ' % a)
                    f.write('\n')
            # print(len(self.q_list[0]))
            # print(self.q_list)
            toc()

        # read q_list from txt file
        self.read_q_list = []
        with open('data/mocap/skate_spin.txt') as f:
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

        # skel.joint("j_abdomen").set_position_upper_limit(10, 0.0)

        self.skeletons[1].set_positions(self.read_q_list[0])

        self.elapsedTime = 0
        self.fn_bvh = 0
        self.flag = 0

        self.contact_force = []
        self.contactPositionLocals = []
        self.bodyIDs = []
        # print("dof: ", skel.ndofs)

    def step(self):
        if self.flag < 10:
            self.skeletons[1].set_positions(self.read_q_list[self.fn_bvh])
            self.flag += 1
        else:
            self.flag = 0
            self.fn_bvh += 1

        # gain_value = 25.0
        gain_value = 50.0

        ndofs = skel.num_dofs()
        h = self.time_step()

        Kp = np.diagflat([0.0] * 6 + [gain_value] * (ndofs - 6))
        Kd = np.diagflat([0.0] * 6 + [2. * (gain_value ** .5)] * (ndofs - 6))
        invM = np.linalg.inv(skel.M + Kd * h)
        p = -Kp.dot(skel.q - self.read_q_list[self.fn_bvh] + skel.dq * h)
        d = -Kd.dot(skel.dq)
        qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        des_accel = p + d + qddot

        ddc = np.zeros(6)

        # print(skel.body('h_blade_left').to_world([0., 0, 0.]), skel.com())

        # HP QP solve

        _ddq, _tau, _bodyIDs, _contactPositions, _contactPositionLocals, _contactForces = hqp.calc_QP(
            skel, des_accel, ddc, 1. / self.time_step())

        del self.contact_force[:]
        del self.bodyIDs[:]
        del self.contactPositionLocals[:]

        self.bodyIDs = _bodyIDs

        for i in range(len(_bodyIDs)):
            skel.body(_bodyIDs[i]).add_ext_force(_contactForces[i], _contactPositionLocals[i])
            self.contact_force.append(_contactForces[i])
            self.contactPositionLocals.append(_contactPositionLocals[i])
        # dartModel.applyPenaltyForce(_bodyIDs, _contactPositionLocals, _contactForces)

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
        del blade_force[:]
        del blade_force_origin[:]
        del rd_footCenter[:]

        com = self.skeletons[0].C
        com[1] = -0.99 + 0.05

        # com = self.skeletons[0].body('h_blade_left').to_world(np.array([0.1040 + 0.0216, +0.80354016 - 0.85354016, -0.054]))
        rd_footCenter.append(com)

        #     blade_force_origin.append(self.skeletons[0].body('h_heel_left').to_world())
        if self.force is not None:
            push_force.append(self.force * 0.05)
            push_force_origin.append(self.skeletons[0].body('h_pelvis').to_world())
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
    q_new[3:6] = q[3:6] - np.array([0., 0.05, 0.])
    skel.set_positions(q_new)

    # pydart.gui.viewer.launch_pyqt5(world)
    viewer = hsv.hpSimpleViewer(viewForceWnd=False)
    # viewer = hsv.hpSimpleViewer(rect=[0, 0, 1280 + 300, 720 + 1 + 55], viewForceWnd=False)

    viewer.doc.addRenderer('controlModel', yr.DartRenderer(world, (255, 255, 255), yr.POLYGON_FILL))
    viewer.doc.addRenderer('ground', yr.DartRenderer(ground, (255, 255, 255), yr.POLYGON_FILL))
    viewer.doc.addRenderer('contactForce', yr.VectorsRenderer(render_vector, render_vector_origin, (255, 0, 0)))
    viewer.doc.addRenderer('pushForce', yr.WideArrowRenderer(push_force, push_force_origin, (0, 255, 0)))
    viewer.doc.addRenderer('rd_footCenter', yr.PointsRenderer(rd_footCenter))
    viewer.doc.addRenderer('ref_motion', yr.JointMotionRenderer(world.motion))

    viewer.motionViewWnd.glWindow.pOnPlaneshadow = (0., -0.98 + 0.0251, 0.)

    viewer.doc.addRenderer('bladeForce', yr.WideArrowRenderer(blade_force, blade_force_origin, (0, 0, 255)))
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
