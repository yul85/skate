from fltk import *
from PyCommon.modules.GUI.hpSimpleViewer import *
from PyCommon.modules.Renderer import ysRenderer as yr
from PyCommon.modules.GUI import ysSimpleViewer_ori as ysvOri
from PyCommon.modules.GUI import ysBaseUI as ybu
from PyCommon.modules.Math import mmMath as mm

import pydart2 as pydart
from SkateUtils.NonHolonomicWorld import NHWorld
from SkateUtils.KeyPoseState import State, revise_pose, IKType
from SkateUtils.DartMotionEdit import DartSkelMotion
import math
import numpy as np
import pickle


class DofObjectInfoWnd(hpObjectInfoWnd):
    def __init__(self, nc, x, y, w, h, doc, dofs):
        super(hpObjectInfoWnd, self).__init__(x, y, w, h, doc)
        self.viewer = None  # type: DofEditingViewer
        self.objectNames.hide()
        self.num_column = nc
        self.column_offset = 0
        self.valObjects = dict()
        self.valObjOffset = 30
        self.states = []  # type: list[State]
        self.num_dof = dofs
        self.root_state = None  # type: State
        self.ref_skel = None  # type: pydart.Skeleton
        self.skmo = None  # type: DartSkelMotion
        self.ori_skmo = None  # type: DartSkelMotion

        self.begin()

        self.start_frame = Fl_Value_Input(250, self.valObjOffset, 40, 20, 's')
        self.start_frame.value(1)
        self.end_frame = Fl_Value_Input(300, self.valObjOffset, 40, 20, 'e')
        self.end_frame.value(300)
        interpolate_btn = fltk.Fl_Button(350, self.valObjOffset, 80, 20, 'interpolate')
        interpolate_btn.callback(self.interpolate)
        offset_btn = fltk.Fl_Button(450, self.valObjOffset, 80, 20, 'offset')
        offset_btn.callback(self.offsetCallback)

        ikd_btn = fltk.Fl_Button(550, self.valObjOffset, 30, 20, 'ik d')
        ikd_btn.callback(self.ikState)
        ikl_btn = fltk.Fl_Button(590, self.valObjOffset, 30, 20, 'ik l')
        ikl_btn.callback(self.ikState)
        ikr_btn = fltk.Fl_Button(630, self.valObjOffset, 30, 20, 'ik r')
        ikr_btn.callback(self.ikState)

        saveBtn = fltk.Fl_Button(680, self.valObjOffset, 80, 20, 'save')
        saveBtn.callback(self.save)
        loadBtn = fltk.Fl_Button(770, self.valObjOffset, 80, 20, 'load')
        loadBtn.callback(self.load)

        self.end()
        self.valObjOffset += 40

    def save(self, ptr):
        file_chooser = fltk.Fl_File_Chooser('.', '*.skmo', 2, 'save key pose file')
        file_chooser.show()
        while file_chooser.shown():
            fltk.Fl.wait()
        if file_chooser.count() == 1:
            filename = file_chooser.value()
            if filename.split('.')[-1] != 'skmo':
                filename += '.skmo'
            if self.skmo is not None:
                self.skmo.save(filename)

    def load(self, ptr):
        file_chooser = fltk.Fl_File_Chooser('.', '*.skmo', FL_SINGLE, 'load key pose file')
        file_chooser.show()
        while file_chooser.shown():
            fltk.Fl.wait()
        if file_chooser.count() == 1:
            self.skmo = DartSkelMotion()
            self.skmo.load(file_chooser.value())
            self.ori_skmo = DartSkelMotion()
            self.ori_skmo.load(file_chooser.value())
            self.viewer.setMaxFrame(len(self.skmo)-1)

    def interpolate(self, ptr):
        if self.skmo is not None:
            start_frame = int(self.start_frame.value())
            end_frame = int(self.end_frame.value())
            for joint_idx in range(self.ref_skel.num_joints()):
                joint = self.ref_skel.joints[joint_idx]  # type: pydart.Joint
                if joint.num_dofs() == 0:
                    continue
                if joint_idx == 0:
                    r_x_idx = self.ref_skel.dof_index(joint.name + '_rot_x')
                    r_y_idx = self.ref_skel.dof_index(joint.name + '_rot_y')
                    r_z_idx = self.ref_skel.dof_index(joint.name + '_rot_z')
                    p_x_idx = self.ref_skel.dof_index(joint.name + '_pos_x')
                    p_y_idx = self.ref_skel.dof_index(joint.name + '_pos_y')
                    p_z_idx = self.ref_skel.dof_index(joint.name + '_pos_z')

                    p_skmo_end = np.asarray([self.skmo.qs[end_frame][p_x_idx],
                                            self.skmo.qs[end_frame][p_y_idx],
                                            self.skmo.qs[end_frame][p_z_idx]])

                    p_ori_skmo_end = np.asarray([self.ori_skmo.qs[end_frame][p_x_idx],
                                                 self.ori_skmo.qs[end_frame][p_y_idx],
                                                 self.ori_skmo.qs[end_frame][p_z_idx]])
                    p_diff = p_skmo_end - p_ori_skmo_end
                    if np.linalg.norm(p_diff) < 0.00001:
                        continue
                    for frame in range(start_frame+1, end_frame):
                        t = (frame-start_frame)/(end_frame - start_frame)
                        self.skmo.qs[frame][p_x_idx] += p_diff[0] * t
                        self.skmo.qs[frame][p_y_idx] += p_diff[1] * t
                        self.skmo.qs[frame][p_z_idx] += p_diff[2] * t

                else:
                    r_x_idx = self.ref_skel.dof_index(joint.name + '_x')
                    r_y_idx = self.ref_skel.dof_index(joint.name + '_y')
                    r_z_idx = self.ref_skel.dof_index(joint.name + '_z')

                R_skmo_end = mm.exp(np.array([self.skmo.qs[end_frame][r_x_idx],
                                              self.skmo.qs[end_frame][r_y_idx],
                                              self.skmo.qs[end_frame][r_z_idx]]))
                R_ori_skmo_end = mm.exp(np.array([self.ori_skmo.qs[end_frame][r_x_idx],
                                                  self.ori_skmo.qs[end_frame][r_y_idx],
                                                  self.ori_skmo.qs[end_frame][r_z_idx]]))
                R_diff = np.dot(R_skmo_end, R_ori_skmo_end.T)
                if np.linalg.norm(R_diff - np.eye(3)) < 0.00001:
                    continue
                for frame in range(start_frame+1, end_frame):
                    t = (frame-start_frame)/(end_frame - start_frame)
                    R_diff_slerp = mm.slerp(np.eye(3), R_diff, t)
                    R_skmo = mm.exp(np.array([self.skmo.qs[frame][r_x_idx],
                                              self.skmo.qs[frame][r_y_idx],
                                              self.skmo.qs[frame][r_z_idx]]))
                    r_skmo_new = mm.logSO3(np.dot(R_diff_slerp, R_skmo))
                    self.skmo.qs[frame][r_x_idx] = r_skmo_new[0]
                    self.skmo.qs[frame][r_y_idx] = r_skmo_new[1]
                    self.skmo.qs[frame][r_z_idx] = r_skmo_new[2]

            for frame in range(start_frame+1, end_frame):
                self.ori_skmo.qs[frame][:] = self.skmo.qs[frame][:]

    def offsetCallback(self, ptr):
        if self.skmo is not None:
            start_frame = int(self.start_frame.value())
            end_frame = int(self.end_frame.value())
            for joint_idx in range(self.ref_skel.num_joints()):
                joint = self.ref_skel.joints[joint_idx]  # type: pydart.Joint
                if joint.num_dofs() == 0:
                    continue
                if joint_idx == 0:
                    r_x_idx = self.ref_skel.dof_index(joint.name + '_rot_x')
                    r_y_idx = self.ref_skel.dof_index(joint.name + '_rot_y')
                    r_z_idx = self.ref_skel.dof_index(joint.name + '_rot_z')
                    p_x_idx = self.ref_skel.dof_index(joint.name + '_pos_x')
                    p_y_idx = self.ref_skel.dof_index(joint.name + '_pos_y')
                    p_z_idx = self.ref_skel.dof_index(joint.name + '_pos_z')

                    p_skmo_end = np.asarray([self.skmo.qs[end_frame][p_x_idx],
                                             self.skmo.qs[end_frame][p_y_idx],
                                             self.skmo.qs[end_frame][p_z_idx]])

                    p_ori_skmo_end = np.asarray([self.ori_skmo.qs[end_frame][p_x_idx],
                                                 self.ori_skmo.qs[end_frame][p_y_idx],
                                                 self.ori_skmo.qs[end_frame][p_z_idx]])
                    p_diff = p_skmo_end - p_ori_skmo_end
                    if np.linalg.norm(p_diff) < 0.00001:
                        continue
                    for frame in range(start_frame+1, end_frame):
                        self.skmo.qs[frame][p_x_idx] += p_diff[0]
                        self.skmo.qs[frame][p_y_idx] += p_diff[1]
                        self.skmo.qs[frame][p_z_idx] += p_diff[2]

                else:
                    r_x_idx = self.ref_skel.dof_index(joint.name + '_x')
                    r_y_idx = self.ref_skel.dof_index(joint.name + '_y')
                    r_z_idx = self.ref_skel.dof_index(joint.name + '_z')

                R_skmo_end = mm.exp(np.array([self.skmo.qs[end_frame][r_x_idx],
                                              self.skmo.qs[end_frame][r_y_idx],
                                              self.skmo.qs[end_frame][r_z_idx]]))
                R_ori_skmo_end = mm.exp(np.array([self.ori_skmo.qs[end_frame][r_x_idx],
                                                  self.ori_skmo.qs[end_frame][r_y_idx],
                                                  self.ori_skmo.qs[end_frame][r_z_idx]]))
                R_diff = np.dot(R_skmo_end, R_ori_skmo_end.T)
                if np.linalg.norm(R_diff - np.eye(3)) < 0.00001:
                    continue
                for frame in range(start_frame+1, end_frame):
                    R_skmo = mm.exp(np.array([self.skmo.qs[frame][r_x_idx],
                                              self.skmo.qs[frame][r_y_idx],
                                              self.skmo.qs[frame][r_z_idx]]))
                    r_skmo_new = mm.logSO3(np.dot(R_diff, R_skmo))
                    self.skmo.qs[frame][r_x_idx] = r_skmo_new[0]
                    self.skmo.qs[frame][r_y_idx] = r_skmo_new[1]
                    self.skmo.qs[frame][r_z_idx] = r_skmo_new[2]

            for frame in range(start_frame+1, end_frame):
                self.ori_skmo.qs[frame][:] = self.skmo.qs[frame][:]

    def ikState(self, ptr):
        ik_type = IKType.DOUBLE
        if ptr.label()[-1] == 'd':
            ik_type = IKType.DOUBLE
        elif ptr.label()[-1] == 'l':
            ik_type = IKType.LEFT
        elif ptr.label()[-1] == 'r':
            ik_type = IKType.RIGHT

        revise_pose(self.ref_skel, None, ik_type, 0.)
        for dof in self.ref_skel.dofs:
            self.valObjects[dof.name].value(ref_skel.q[dof.name])
        self.viewer.motionViewWnd.glWindow.redraw()
        self.redraw()

    def add1DSlider(self, name, minVal, maxVal, valStep, initVal):
        self.begin()
        slider = Fl_Hor_Value_Slider(self.viewer.panelWidth * self.column_offset + 10, self.valObjOffset, self.viewer.panelWidth - 30, 18, name)
        slider.textsize(8)
        slider.bounds(minVal, maxVal)
        slider.value(initVal)
        slider.step(valStep)
        slider.label(name)
        slider.name = name
        self.end()
        self.addValObjects(slider)
        self.valObjOffset += 40
        if self.valObjOffset + 40 > self.h():
            self.column_offset += 1
            self.valObjOffset = 70

        return slider

    def addValObjects(self, obj):
        self.valObjects[obj.name] = obj


class DofEditingViewer(hpSimpleViewer):
    def __init__(self, rect=None, title='hpSimpleViewer', panelWidth=300, numColumn=1, dofs=60):
        ybu.BaseWnd.__init__(self, rect, title, ysvOri.SimpleSetting())
        self.title = title
        self.doc = ysvOri.SimpleDoc()
        self.begin()
        self.panelWidth = panelWidth
        t = .1
        self.renderersWnd = ysvOri.RenderersWnd(self.w() - numColumn * panelWidth, 0, numColumn * panelWidth, int(self.h()*t), self.doc)
        self.objectInfoWnd = DofObjectInfoWnd(numColumn, self.w() - numColumn * panelWidth, int(self.h()*t), numColumn * panelWidth, int(self.h()*(1-t)), self.doc, dofs)
        self.motionViewWnd = hpMotionViewWnd(0, 0, self.w() - numColumn * panelWidth, self.h(), self.doc)
        self.end()
        self.resizable(self.motionViewWnd)
        self.size_range(600, 400)

        self.motionViewWnd.cForceWnd = None
        self.objectInfoWnd.viewer = self


if __name__ == '__main__':
    print('Example: Skating -- pushing side to side')

    pydart.init()
    print('pydart initialization OK')

    ref_world = NHWorld(1./1200., '../data/skel/skater_3dof_with_ground.skel')
    print('World OK')

    ref_skel = ref_world.skeletons[1]

    viewer_w, viewer_h = 900, 1200
    viewer = DofEditingViewer(rect=(0, 0, viewer_w + 900, 1 + viewer_h + 55), panelWidth=300, numColumn=3, dofs=ref_skel.num_dofs())
    viewer.record(False)
    viewer.objectInfoWnd.ref_skel = ref_world.skeletons[1]

    rd_ext_force_vec = []
    rd_ext_force_ori = []
    rd_COM = []

    viewer.doc.addRenderer('MotionModel', yr.DartRenderer(ref_world, (194,207,245), yr.POLYGON_FILL))
    viewer.doc.addRenderer('COM', yr.PointsRenderer(rd_COM, (255, 0, 0)))

    viewer.startTimer(1/30.)

    sliders = {}
    dof_names = []

    viewer.setMaxFrame(1000)

    def onSliderChange(ptr):
        _skel = ref_world.skeletons[1]
        _q = np.asarray([sliders[dof_name].value() for dof_name in dof_names])
        _skel.set_positions(_q)
        if viewer.objectInfoWnd.skmo is not None:
            viewer.objectInfoWnd.skmo.qs[viewer.motionViewWnd.getCurrentFrame()] = _q
        viewer.motionViewWnd.glWindow.redraw()

    for dof in ref_skel.dofs:
        max_val = 3.
        if '_pos_' in dof.name:
            max_val = 20.
        slider = viewer.objectInfoWnd.add1DSlider(dof.name, -max_val, max_val, 0.001, 0.)
        slider.callback(onSliderChange)
        sliders[dof.name] = slider
        dof_names.append(dof.name)

    time = [0.]

    viewer.motionViewWnd.panel.last.hide()
    viewer.motionViewWnd.panel.prev.hide()
    viewer.motionViewWnd.panel.next.hide()

    def preCallback(frame):
        if viewer.objectInfoWnd.skmo is not None:
            ref_skel.set_positions(viewer.objectInfoWnd.skmo.get_q(frame))
            del rd_COM[:]
            com = ref_skel.com()
            com[1] = 0.
            rd_COM.append(com)
            print(ref_skel.com()[1])
        for dof in ref_skel.dofs:
            sliders[dof.name].value(ref_skel.q[dof.name])

    viewer.setPreFrameCallback_Always(preCallback)
    viewer.show()

    Fl.run()
