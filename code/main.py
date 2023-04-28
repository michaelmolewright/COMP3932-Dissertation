
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import helper as util
import twoMoonsSeg as tm
import imageSeg as img
import buildGraph as bd
import segmentation as sgm
from sklearn.datasets import make_moons

import imageSeg as mk
import random
import misc

#mk.segment_image('../images/plane.jpg','../images/planeDesired.jpg')
#mk.segment_image_tester('../images/polarBear.jpg',50, '../images/polarBear.jpg')

#img = subsample('../images/hill.jpg', 40)
#tester(img)

#vals = [[1,2],[1,4],[1,6],[1,8],[1,10]]
#j = [40]
#tm.experiment(1000,0.15,3,vals)
#tm.experiment(1000,0.175, 5,vals)
#tm.experiment(1000,0.2, 5,vals)

#misc.plot_gausian_func()

mk.plot_image_eigenfunctions('../images/polarBear.jpg')

