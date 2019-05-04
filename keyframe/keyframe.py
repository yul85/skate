from fltk import *
from PyCommon.modules.GUI.hpSimpleViewer import *
from PyCommon.modules.Renderer import ysRenderer as yr
from PyCommon.modules.GUI import ysSimpleViewer_ori as ysvOri
from PyCommon.modules.GUI import ysBaseUI as ybu

import pydart2 as pydart
from SkateUtils.NonHolonomicWorld import NHWorld
from SkateUtils.KeyPoseState import State, revise_pose
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
        self.skel = None  # type: pydart.Skeleton
        self.ref_skel = None  # type: pydart.Skeleton

        self.begin()

        dt_slider = Fl_Hor_Value_Slider(370, self.valObjOffset, 170, 20, 'dt')
        dt_slider.textsize(8)
        dt_slider.bounds(0., 10.)
        dt_slider.value(1.)
        dt_slider.step(0.01)
        dt_slider.label('dt')
        dt_slider.name = 'dt'
        self.addValObjects(dt_slider)
        dt_slider.callback(self.onChangeDtSlider)

        self.state_name_choice = Fl_Choice(5, 5, w-10, 20, '')
        self.state_name_choice.callback(self.changeState)
        self.next_state_choice = Fl_Choice(280, self.valObjOffset, 80, 20, '')
        self.next_state_choice.add('None')
        self.next_state_choice.callback(self.changeNextState)
        self.addState(None)

        add_state_btn = Fl_Button(10, self.valObjOffset, 80, 20, 'add state')
        add_state_btn.callback(self.addState)
        del_state_btn = Fl_Button(100, self.valObjOffset, 80, 20, 'del state')
        del_state_btn.callback(self.delState)
        renew_name_btn = Fl_Button(190, self.valObjOffset, 80, 20, 'renew name')
        del_state_btn.hide()
        renew_name_btn.hide()

        ik_btn = fltk.Fl_Button(550, self.valObjOffset, 80, 20, 'ik ground')
        ik_btn.callback(self.ikState)

        saveBtn = fltk.Fl_Button(640, self.valObjOffset, 80, 20, 'save')
        saveBtn.callback(self.save)
        loadBtn = fltk.Fl_Button(730, self.valObjOffset, 80, 20, 'load')
        loadBtn.callback(self.load)

        self.end()
        self.valObjOffset += 40

    def save(self, ptr):
        file_chooser = fltk.Fl_File_Chooser('.', '*.skkey', 2, 'save key pose file')
        file_chooser.show()
        while file_chooser.shown():
            fltk.Fl.wait()
        if file_chooser.count() == 1:
            filename = file_chooser.value()
            if filename.split('.')[-1] != 'skkey':
                filename += '.skkey'
            with open(filename, 'wb') as f:
                pickle.dump(self.states, f)

    def load(self, ptr):
        file_chooser = fltk.Fl_File_Chooser('.', '*.skkey', FL_SINGLE, 'load key pose file')
        file_chooser.show()
        while file_chooser.shown():
            fltk.Fl.wait()
        if file_chooser.count() == 1:
            with open(file_chooser.value(), 'rb') as f:
                self.states = pickle.load(f)  # type: list[State]
                self.state_name_choice.clear()
                self.next_state_choice.clear()

                for _state in self.states:
                    self.state_name_choice.add(_state.name)
                    self.next_state_choice.add(_state.name)
                self.next_state_choice.add('None')
                self.state_name_choice.value(0)
                self.root_state = self.states[0]
                self.changeState(self.state_name_choice)

    def onChangeDtSlider(self, ptr):
        selected_state = self.getSelectedState()
        selected_state.dt = ptr.value()

    def getSelectedState(self):
        return self.states[self.state_name_choice.value()]

    def addState(self, ptr):
        new_state_name = 'State' + str(len(self.states))
        self.next_state_choice.remove(len(self.states))
        self.state_name_choice.add(new_state_name)
        self.next_state_choice.add(new_state_name)
        self.next_state_choice.add('None')
        self.state_name_choice.value(len(self.states))
        self.next_state_choice.value(len(self.states)+1)
        self.states.append(State(new_state_name, 1., 0., 0., np.zeros(self.num_dof)))
        self.valObjects['dt'].value(1.)

        if len(self.states) == 1:
            self.root_state = self.states[0]
        else:
            self.states[-2].set_next(self.states[-1])

        if self.ref_skel is not None:
            self.ref_skel.set_positions(self.states[-1].angles)
            self.viewer.motionViewWnd.glWindow.redraw()

    def delState(self, ptr):
        pass

    def changeState(self, ptr):
        """

        :param ptr:
        :type ptr: Fl_Choice
        :return:
        """
        selected_state = self.states[self.state_name_choice.value()]
        for i in range(self.num_dof):
            self.valObjects[self.skel.dof(i).name].value(selected_state.angles[i])

        if selected_state.next is None:
            self.next_state_choice.value(self.next_state_choice.find_index('None'))
        else:
            self.next_state_choice.value(self.next_state_choice.find_index(selected_state.next.name))

        self.valObjects['dt'].value(selected_state.dt)
        self.ref_skel.set_positions(selected_state.angles)
        self.viewer.motionViewWnd.glWindow.redraw()

    def changeNextState(self, ptr):
        """

        :param ptr:
        :type ptr: Fl_Choice
        :return:
        """
        selected_state = self.getSelectedState()
        if ptr.value() == len(self.states):
            selected_state.set_next(None)
        else:
            selected_state.set_next(self.states[ptr.value()])

    def ikState(self, ptr):
        revise_pose(self.ref_skel, self.getSelectedState())
        self.changeState(self.state_name_choice)

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

    world = NHWorld(1./1200., '../data/skel/skater_3dof_with_ground.skel')
    ref_world = NHWorld(1./1200., '../data/skel/skater_3dof_with_ground.skel')
    print('World OK')

    skel = world.skeletons[1]
    print('skeleton position OK')

    viewer_w, viewer_h = 900, 1200
    viewer = DofEditingViewer(rect=(0, 0, viewer_w + 900, 1 + viewer_h + 55), panelWidth=300, numColumn=3, dofs=skel.num_dofs())
    viewer.record(False)
    viewer.setMaxFrame(1000)
    viewer.objectInfoWnd.skel = skel
    viewer.objectInfoWnd.ref_skel = ref_world.skeletons[1]

    viewer.doc.addRenderer('MotionModel', yr.DartRenderer(ref_world, (194,207,245), yr.POLYGON_FILL))
    viewer.doc.addRenderer('controlModel', yr.DartRenderer(world, (255,255,255), yr.POLYGON_FILL))

    viewer.startTimer(1/30.)

    sliders = {}
    dof_names = []

    def onSliderChange(ptr):
        _skel = ref_world.skeletons[1]
        _q = np.asarray([sliders[dof_name].value() for dof_name in dof_names])
        _skel.set_positions(_q)
        selected_state = viewer.objectInfoWnd.getSelectedState()
        selected_state.angles = _q
        viewer.motionViewWnd.glWindow.redraw()

    for dof in skel.dofs:
        slider = viewer.objectInfoWnd.add1DSlider(dof.name, -3., 3., 0.001, 0.)
        slider.callback(onSliderChange)
        sliders[dof.name] = slider
        dof_names.append(dof.name)

    state = [viewer.objectInfoWnd.root_state]
    time = [0.]

    def onClickFirst(ptr):
        viewer.motionViewWnd.pause()
        world.reset()
        viewer.motionViewWnd.goToFrame(-1)
        viewer.objectInfoWnd.state_name_choice.value(0)
        state[0] = viewer.objectInfoWnd.root_state
        ref_world.skeletons[1].set_positions(state[0].angles)
        viewer.motionViewWnd.glWindow.redraw()
        time[0] = 0.

    viewer.motionViewWnd.panel.first.callback(onClickFirst)
    Kp = 400.
    Kd = 40.
    h = world.time_step()

    def simulateCallback(frame):
        if frame == 0:
            state[0] = viewer.objectInfoWnd.root_state
            skel.set_positions(state[0].angles)
        for i in range(40):
            skel.set_forces(skel.get_spd(state[0].angles, h, Kp, Kd))
            world.step()
            time[0] += h
            if time[0] >= state[0].dt and state[0].next is not None:
                time[0] -= state[0].dt
                prev_state_name = state[0].name
                state[0] = state[0].next
                ref_world.skeletons[1].set_positions(state[0].angles)

                viewer.objectInfoWnd.state_name_choice.value(viewer.objectInfoWnd.state_name_choice.find_index(state[0].name))
                viewer.objectInfoWnd.changeState(viewer.objectInfoWnd.state_name_choice)
                print('Transition: ', prev_state_name, ' -> ', state[0].name)

    viewer.setSimulateCallback(simulateCallback)
    viewer.show()

    Fl.run()
