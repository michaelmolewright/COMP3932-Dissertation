
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

#matplotlib.use('Agg')

#twoMoon.test_harness()


#twoMoon.run_test_with_plot(1000, 0.1, 1.33, 1.5, 500, "./plots/GL_Seg1.png")
#twoMoon.run_test_with_plot(1000, 0.1, 0.2, 10, 500, "./plots/GL_Seg2.png")
#twoMoon.run_test_with_plot(1000, 0.1, 1.33, 1.5, 500, "./plots/GL_Seg3.png")

#twoMoon.playground(1000, 50)

#util.plot_two_moons(500, 0, "./plots/test1.png")
#util.plot_two_moons(1000, 0.05, "./plots/test2.png")
#util.plot_two_moons(1000, 0.15, "./plots/test3.png")
#img.messing()

#X, Y = make_moons(n_samples=1000, noise=0.1)


def tester(img):

    pixels = list(img.getdata())
    w, h = img.size
    gb = bd.graphBuilder()
    gb.setup(pixels)
    segmenter = sgm.segment()
    segmenter.setup(gb.graph)
    for i in range(20):
        x1 = random.random() * 3
        x2 = random.random() * 20
        x3 = random.random() * 10
        x4 = random.randint(5,20)
        
        
        
        gb.gaussian_image(w, x4, x1, x2)
        segmenter.setup(gb.graph)
        #seg = segmenter.gl_method(0.1, 2, x3, 500, 20)
        seg = segmenter.fielder_method()
        mk.make_seg_img('./output' + str(i) + '.png', w, h, seg)
        #print('output', str(i), '.png -- r=',x4, ', sig_i=',x1,', sig_x=',x2, ',eps=', x3 )


#mk.segment_image_tester('../images/horse_test.jpg', 10)

#img = subsample('../images/hill.jpg', 40)
#tester(img)

vals = [2,5,7,10,20]
tm.experiment(1000,0.15,1,vals)
#tm.experiment(1000,0.175, 10,vals)
#tm.experiment(1000,0.2, 10,vals)

#variables = ["Gaussian Function", "Local Scaling", "Cosine" ]
#values = [2.464890571, 5.161708067, 3.374487338]
#util.plot_bar(variables, values, "speedBar")