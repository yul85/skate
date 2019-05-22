from fltk import *
from PyCommon.modules.GUI import ysSimpleViewer_ori as ysv
from PyCommon.modules.Renderer import ysRenderer as yr
from PyCommon.modules.Motion.ysMotionLoader import readBvhFile


def main(bvhFilePath):
    motion = readBvhFile(bvhFilePath, .01)
    motion2 = readBvhFile(bvhFilePath, .01, True)
    print(motion[0].skeleton)

    viewer = ysv.SimpleViewer()
    viewer.record(False)
    viewer.doc.addRenderer('motion', yr.JointMotionRenderer(motion, (0,255,0)))
    viewer.doc.addObject('motion', motion)
    viewer.doc.addRenderer('motion2', yr.JointMotionRenderer(motion2, (255,0,0)))
    viewer.doc.addObject('motion2', motion2)

    viewer.startTimer(1/motion.fps)
    viewer.show()

    Fl.run()


if __name__ == '__main__':
    main('../skate_cma/jump_back.bvh')