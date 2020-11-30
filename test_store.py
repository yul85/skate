import numpy as np
import os
from datetime import datetime

q_box = []
for i in range(10):
    q_box.append(np.array([0, 1, 2]))

day = datetime.today().strftime("%Y%m%d%H%M%S")

newpath = 'OptRes'
if not os.path.exists(newpath):
    os.makedirs(newpath)

with open('OptRes/opt_'+day+'.txt', 'w') as f:
    for item in q_box:
        f.write("%s\n" % item)

