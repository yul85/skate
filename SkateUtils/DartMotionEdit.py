import pydart2 as pydart
import numpy as np
import copy


class DartSkelMotion(object):
    def __init__(self):
        self.qs = []
        self.dqs = []
        self.fps = 30.

    def __len__(self):
        assert len(self.qs) == len(self.dqs)
        return len(self.qs)

    def load(self, filename):
        with open(filename, 'r') as f:
            self.fps = 1./float(f.readline().split(' ')[-1])
            for s in f.read().splitlines():
                ss = s.replace('[', '').split(']')
                sq, sdq = list(map(float, ss[0].split(','))), list(map(float, ss[1].split(',')))
                self.qs.append(np.asarray(sq))
                self.dqs.append(np.asarray(sdq))

    def save(self, filename):
        with open(filename, 'w') as f:
            for frame in range(len(self.qs)):
                f.write(str([d for d in np.asarray(self.qs[frame])]))
                f.write(str([d for d in np.asarray(self.dqs[frame])]))
                f.write('\n')

    def append(self, _q, _dq):
        self.qs.append(copy.deepcopy(_q))
        self.dqs.append(copy.deepcopy(_dq))

    def extend(self, frame, _qs, _dqs):
        qs = copy.deepcopy(_qs)
        dqs = copy.deepcopy(_dqs)
        offset_x = self.qs[frame][3] - qs[0][3]
        offset_z = self.qs[frame][5] - qs[0][5]
        del self.qs[frame:]
        del self.dqs[frame:]

        self.qs.extend(qs)
        self.dqs.extend(dqs)

        for i in range(len(qs)):
            self.qs[frame+i][3] += offset_x
            self.qs[frame+i][5] += offset_z

    def translate_by_offset(self, offset):
        self.qs[3] += offset[0]
        self.qs[5] += offset[2]

    def get_q(self, frame):
        return self.qs[frame]

    def get_dq(self, frame):
        return self.dqs[frame]


if __name__ == '__main__':
    mo = DartSkelMotion()
    mo.load('skate.skmo')
