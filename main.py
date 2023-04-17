
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import helper as util
import twoMoonsSeg as twoMoon
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


def tester():
    im = Image.open("./images/banana.jpg")
    pixels = list(im.getdata())
    w, h = im.size
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
        mk.make_seg_img('./plots/output' + str(i) + '.png', w, h, seg)
        print('output', str(i), '.png -- r=',x4, ', sig_i=',x1,', sig_x=',x2, ',eps=', x3 )



def seg_image():
    im = Image.open("output_image.jpg")
    pixels = list(im.getdata())
    w, h = im.size
    gb = bd.graphBuilder()
    gb.setup(pixels)
    segmenter = sgm.segment()
    segmenter.setup(gb.graph)
    gb.gaussian_image(w, 12, 2, 18, gtype="colour")
    segmenter.setup(gb.graph)
    
    seg = segmenter.gl_method(0.1, 2, 2.9, 500, 20)
    mk.make_seg_img('./plots/gl_method.png', w, h, seg)

    seg = segmenter.fielder_method()
    mk.make_seg_img('./plots/fielder_method.png', w, h, seg)

    seg = segmenter.perona_freeman_method(8)
    mk.make_seg_img('./plots/perona_freeman_method.png', w, h, seg)



def subsample(img_path, factor):
    """
    Subsamples an image by averaging the pixels in each block of size `factor x factor`.
    
    Args:
        img_path (str): The path to the input image.
        factor (int): The subsampling factor. A factor of 2 will reduce the image to half its size,
            a factor of 3 will reduce it to a third, and so on.
    
    Returns:
        A new PIL Image object representing the subsampled image.
    """
    with Image.open(img_path) as img:
        width, height = img.size
        
        # Calculate the new size of the subsampled image.
        new_width = width // factor
        new_height = height // factor
        
        subsampled = Image.new('RGB', (new_width, new_height))

        # Loop over each block of size `factor x factor` in the input image.
        for y in range(0, height, factor):
            for x in range(0, width, factor):
                
                block = img.crop((x, y, x + factor, y + factor))
                
                r, g, b = block.resize((1, 1)).getpixel((0, 0))

                if x//factor < new_width and y//factor < new_height:
                    subsampled.putpixel( (x//factor, y//factor), (r, g, b))
                
     
        return subsampled


#fil = subsample("hill.jpg", 40)

#fil.save('output_image.jpg')

#seg_image()

def remake_seg_image(original, segmentation, factor):
    widthO, heightO = original.size
    widthS, heightS = segmentation.size

    final = Image.new('RGB', (widthO, heightO))

    segX = 0
    segY = 0
    for y in range(0, heightO):
        segX = 0
        for x in range(0, widthO):

            if segX < widthS and segY < heightS:
                segPix = segmentation.getpixel( (segX, segY) )

            if segPix == (255,255,255):
                final.putpixel( (x,y), original.getpixel( (x,y) ) )
            else:
                final.putpixel( (x,y), (0,0,0) )

            if x % factor == (factor - 1):
                segX += 1
                if segX < widthS and segY < heightS:
                    print(segX, segY, segmentation.getpixel( (segX, segY) ))

        if y % factor == (factor - 1):
            segY += 1
    return final


segged = Image.open("./plots/gl_method.png")
orig = Image.open("hill.jpg")
fin = remake_seg_image(orig, segged, 40)

fin.save('output_image.jpg')