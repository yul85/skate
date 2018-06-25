from . import ysSimpleViewer_ori as ysvOri
from ..GUI import ysBaseUI as ybu
import fltk
try:
    # for python3
    import pickle
except:
    # for python2.7
    import cPickle as pickle


class hpSimpleViewer(ysvOri.SimpleViewer):
    def __init__(self, rect=None, title='hpSimpleViewer', viewForceWnd=True):
        ybu.BaseWnd.__init__(self, rect, title, ysvOri.SimpleSetting())
        self.title = title
        self.doc = ysvOri.SimpleDoc()
        self.begin()
        self.panelWidth = 300
        panelWidth = self.panelWidth
        cForceHeight = 200
        t = .2
        self.renderersWnd = ysvOri.RenderersWnd(self.w()-panelWidth, 0, panelWidth, int(self.h()*t), self.doc)
        self.objectInfoWnd = hpObjectInfoWnd(self.w()-panelWidth, int(self.h()*t), panelWidth, int(self.h()*(1-t)), self.doc)
        self.motionViewWnd = None #type: hpMotionViewWnd
        self.cForceWnd = None #type: None|hpContactForceGraphWnd
        if viewForceWnd:
            self.motionViewWnd = hpMotionViewWnd(0, 0, self.w()-panelWidth, self.h()-cForceHeight, self.doc)
            self.cForceWnd = hpContactForceGraphWnd(0, self.h()-cForceHeight, self.w()-panelWidth-40, cForceHeight, self.doc)
        else:
            self.motionViewWnd = hpMotionViewWnd(0, 0, self.w()-panelWidth, self.h(), self.doc)
        self.end()
        self.resizable(self.motionViewWnd)
        # self.resizable(self.cForceWnd)
        self.size_range(600, 400)

        if viewForceWnd:
            self.cForceWnd.viewer = self
            self.motionViewWnd.cForceWnd = self.cForceWnd
        else:
            self.motionViewWnd.cForceWnd = None
        self.objectInfoWnd.viewer = self


class hpMotionViewWnd(ysvOri.MotionViewWnd):
    def __init__(self, x, y, w, h, doc):
        ysvOri.MotionViewWnd.__init__(self, x, y, w, h, doc)
        self.mov = False

    def goToFrame(self, frame):
        super(hpMotionViewWnd, self).goToFrame(frame)
        if self.cForceWnd is not None:
            self.cForceWnd.redraw()


class hpObjectInfoWnd(ysvOri.ObjectInfoWnd):
    def __init__(self, x, y, w, h, doc):
        super(hpObjectInfoWnd, self).__init__(x, y, w, h, doc)
        self.valObjects = dict()
        self.valObjOffset = 30

        self.begin()
        saveBtn = fltk.Fl_Button(10, self.valObjOffset, 80, 20, 'param save')
        saveBtn.callback(self.save)
        loadBtn = fltk.Fl_Button(100, self.valObjOffset, 80, 20, 'param load')
        loadBtn.callback(self.load)
        self.end()
        self.valObjOffset += 40

        # super(hpObjectInfoWnd, self).__init__(x, y, w, h, doc)

    def update(self, ev, doc):
        super(hpObjectInfoWnd, self).update(ev, doc)

    def addValObjects(self, obj):
        self.valObjects[obj.name] = obj
        pass

    def getValobject(self, name):
        return self.valObjects[name]
        pass

    def getValObjects(self):
        return self.valObjects.values()

    def getVals(self):
        return (v.value() for v in self.valObjects.values())

    def getVal(self, name):
        try:
            return self.valObjects[name].value()
        except Exception as e:
            print(e)
            return 0

    def getNameAndVals(self):
        objValDict = dict()
        for k, v in self.valObjects.items():
            objValDict[k] = v.value()
        return objValDict

    def setVal(self, name, val):
        try:
            self.valObjects[name].value(val)
        except Exception as e:
            print(e)

    def addBtn(self, name, callback):
        self.begin()
        btn = fltk.Fl_Button(10, self.valObjOffset, 80, 20, name)
        btn.callback(callback)
        self.end()
        self.valObjOffset += 40

    def add1DSlider(self, name, minVal, maxVal, valStep, initVal):
        self.begin()
        slider = fltk.Fl_Hor_Value_Slider(10, self.valObjOffset, self.viewer.panelWidth - 30, 18, name)
        slider.textsize(8)
        slider.bounds(minVal, maxVal)
        slider.value(initVal)
        slider.step(valStep)
        slider.label(name)
        slider.name = name
        self.end()
        self.addValObjects(slider)
        self.valObjOffset += 40

    def add1DRoller(self, name):
        class hpRoller(fltk.Fl_Roller):
            def handle(self, event):
                if self.handler is not None:
                    self.handler(self, event)
                return super(hpRoller, self).handle(event)
            def set_handler(self, handler):
                self.handler = handler


        self.begin()
        roller = hpRoller(10, self.valObjOffset, self.viewer.panelWidth - 30, 18, name)
        roller.type(fltk.FL_HORIZONTAL)
        roller.bounds(-1., 1.)
        roller.value(0.)
        roller.step(0.001)
        roller.label(name)
        roller.handler = None
        roller.name = name
        self.end()
        self.addValObjects(roller)
        self.valObjOffset += 40


    def add3DSlider(self, name, minVal, maxVal, valStep, initVal):
        self.begin()

        self.end()
        pass

    def save(self, obj):
        f = open(self.viewer.title+'.param', 'wb')
        pickle.dump(self.getNameAndVals(), f)
        f.close()

    def load(self, obj):
        filefile = fltk.Fl_File_Chooser('.', '*.param', fltk.FL_SINGLE, 'load parameter file')
        filefile.show()
        while filefile.shown():
            fltk.Fl.wait()
        if filefile.count() == 1:
            # f = file(self.viewer.title+'param', 'r')
            f = open(filefile.value(), 'rb')
            objVals = pickle.load(f)
            f.close()
            for k, v in objVals.iteritems():
                if k in self.valObjects.keys():
                    self.valObjects[k].value(v)



class hpContactForceGraphWnd(fltk.Fl_Widget, ybu.Observer):
    def __init__(self, x, y, w, h, doc):
        self.doc = doc
        self.doc.attach(self)
        super(hpContactForceGraphWnd, self).__init__(x, y, w, h)
        self.data = []
        self.dataLength = 0
        self.dataSetName = []
        self.dataSetColor = []

        self.dataCheckBtn = []
        self.dataCheckBtnOffset = 0

        self.curFrame = -1
        self.viewer = None

        self.zoomslider = fltk.Fl_Hor_Value_Slider(self.x()+self.w() - 280, self.y()+20, 250, 18, "zoom")
        self.zoomslider.bounds(1, 10)
        self.zoomslider.value(1)
        self.zoomslider.step(1)
        self.zoomslider.callback(self.checkBtnCallback)

        self.yzoomslider = fltk.Fl_Hor_Value_Slider(self.x()+self.w() - 280, self.y()+60, 250, 18, "y zoom")
        self.yzoomslider.bounds(1, 20)
        self.yzoomslider.value(1)
        self.yzoomslider.step(1)
        self.yzoomslider.callback(self.checkBtnCallback)

    def update(self, ev, doc):
        if ev == ysvOri.EV_addObject:
            self.dataLength = doc.motionSystem.getMaxFrame()
        self.redraw()

    def addDataSet(self, name, color):
        self.data.append([0.] * (self.dataLength+1))
        self.dataSetName.append(name)
        self.dataSetColor.append(color)
        checkBox = fltk.Fl_Check_Button(self.x(), self.y() + self.dataCheckBtnOffset, 30, 40, name)
        checkBox.value(True)
        checkBox.callback(self.checkBtnCallback)
        self.dataCheckBtn.append(checkBox)
        self.dataCheckBtnOffset += 40
        self.redraw()

    def addData(self, name, val):
        dataIdx = self.dataSetName.index(name)
        self.data[dataIdx].append(val)
        self.redraw()

    def insertData(self, name, valIdx, val):
        try:
            dataIdx = self.dataSetName.index(name)
            self.data[dataIdx][valIdx] = val
            self.redraw()
        except ValueError:
            print("error")
            pass

    def draw(self):
        fltk.fl_draw_box(fltk.FL_FLAT_BOX, 40+self.x(), self.y(), self.w()-40, self.h(), fltk.fl_rgb_color(192, 192, 192))
        ratio = float(self.w()-40)/self.dataLength
        ratio *= self.zoomslider.value()
        dataRatio = 1./self.yzoomslider.value()
        for dataIdx in range(len(self.data)):
            if self.dataCheckBtn[dataIdx].value():
                for valIdx in range(1, self.dataLength-1):
                    fltk.fl_color(self.dataSetColor[dataIdx])
                    fltk.fl_line(40+self.x()+int(ratio * (valIdx-1)), int(self.y()+self.h() - self.data[dataIdx][valIdx-1]*dataRatio)-3,
                                 40+self.x()+int(ratio * valIdx), int(self.y()+self.h() - self.data[dataIdx][valIdx]*dataRatio)-3)

        frame = self.viewer.getCurrentFrame()
        if frame > -1:
            fltk.fl_color(fltk.FL_BLUE)
            fltk.fl_line(40+self.x()+int(ratio * frame), int(self.y()+self.h())-3,
                         40+self.x()+int(ratio * frame), int(self.y()-3))

        self.zoomslider.redraw()
        self.yzoomslider.redraw()

    def checkBtnCallback(self, ptr):
        self.redraw()
        pass


