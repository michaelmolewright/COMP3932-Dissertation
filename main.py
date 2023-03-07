
import matplotlib.pyplot as plt
import matplotlib

import helper as util
import twoMoonsSeg as twoMoon

matplotlib.use('Agg')

twoMoon.test_harness()


''' HOW TO MAKE THE VISUALISATION
new_x = []
new_y = []

for r in X:
    new_x.append(r[0])
    new_y.append(r[1])

print(end - start," -- 6. Segmentation")
plt.scatter(new_x, new_y, c=seg)
plt.savefig("moons.png")
'''