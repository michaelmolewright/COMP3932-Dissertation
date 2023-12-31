from PIL import Image
import networkx as nx
import helper as util
import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import eigsh
import random

import buildGraph as bd
import segmentation as sgm


def imageTesting():

    im = Image.open("./images/banana.jpg")
    pixels = list(im.getdata())
    w, h = im.size
    
    print(len(pixels))
    #G = create_graph(pixels, w, h)
    G = create_graph_fully_connected(pixels, w, h, 10, 0.01, 4)
    
    print("graph made")
    Lnorm = nx.normalized_laplacian_matrix(G)
    #affinity = nx.adjacency_matrix(G, weight="weight")
    #aff_norm = normalize(affinity, norm="l1")
    
    print("normalized_laplacian_matrix made")
    eigvalues, eigVectors = np.linalg.eigh(Lnorm.A)


    
    print("eigvalues, eigVectors made")
    eigens = util.sortEigens(eigvalues, eigVectors)
    #E,V = eigsh(Lnorm, 4, which = 'SM')
    #E = E[:,np.newaxis]
    Eval = np.asarray(eigens["values"][:20])
    Evec = np.asarray(eigens["vectors"][:20])
    Evec = Evec.T
    Eval = Eval[:,np.newaxis]
    
    print("sortEigens done")


    seg1 = util.second_eigenvector_segmentation(eigens["vectors"][1])
    nodes = []
    for i,val in enumerate(pixels):
        nodes.append(i)

    print("second_eigenvector_segmentation done")
    seg2 = util.ginzburg_landau_segmentation_two(nodes, Eval, Evec, 0.1, 1, 2, 500)
    print("ginzburg_landau_segmentation done")
    seg3 = util.shi_malek_segmentation(10, eigens["vectors"])
    print("shi_malek_segmentation done")
    seg4 = util.perona_freeman_segmentation(10, eigens["vectors"])

    make_seg_img('output1.png', w, h, seg1)
    make_seg_img('output2.png', w, h, seg2)
    make_seg_img('output3.png', w, h, seg3)
    make_seg_img('output4.png', w, h, seg4)
    
def create_graph(pixels, w, h):
    G = nx.Graph()
    for x in range(w):
        for y in range(h):
            G.add_node( (x,y) )

    directions = [(x-1,y), (x+1,y), (x,y+1), (x,y-1)]
    for x in range(w):

        for y in range(h):
            node = (x,y)
            for dir in directions:
                if dir[0] >= 0 and dir[0] < w and dir[1] >= 0 and dir[1] < h:
                    if is_adjancent(node, dir):
                        if not G.has_edge(node, dir):
                            RGB1 = pixels[node[1] * w + node[0]]
                            RGB2 = pixels[dir[1] * w + dir[0]]
                            d = get_dist(RGB1[0]-RGB2[0], RGB1[1]-RGB2[1], RGB1[2]-RGB2[2])
                            G.add_edge(node, dir, weight= 2.781 ** ( -((d ** 2)) / 2 ))
    return G

def is_adjancent(pixel1, pixel2):
    if abs(pixel1[0] - pixel2[0]) <= 1 and  abs(pixel1[1] - pixel2[1]) <= 1:
        return True
    return False

def get_dist(a, b, c):
    return (a**2 + b**2 + c**2)**0.5

def create_graph_fully_connected(pixels, w, h, r, sig_i, sig_x):
    G = nx.Graph()
    for i, val in enumerate(pixels):
            G.add_node( i )
    
    print(len(pixels))
    for i, RGB1 in enumerate(pixels):
        for j, RGB2 in enumerate(pixels):
            if j > i:
                #d = get_dist(RGB1[0]-RGB2[0], RGB1[1]-RGB2[1], RGB1[2]-RGB2[2])
                x1 = i % w
                y1 = i // w
                x2 = j % w
                y2= j // w

                pixels_away = ( (x2-x1)**2 + (y2-y1)**2 )**0.5

                if pixels_away < r:
                    I1 = (RGB1[0] + RGB1[1] + RGB1[2]) /3
                    I2 = (RGB2[0] + RGB2[1] + RGB2[2]) /3

                    new_d = I1 - I2
                    distanceVal = (pixels_away ** 2) / sig_x
                    colourVal = 2.781 ** ( -((new_d ** 2)) / sig_i )
                    G.add_edge(i, j, weight = colourVal * distanceVal )
        if i %100 == 0:
            print(i)
    return G

'''
# Get the width and height of the image
width, height = im.size

# Accessing the RGB value of the pixel at coordinates (x, y)
x = 0
y = 0
r, g, b = pixels[y * width + x]

'''

def make_seg_img(path, w, h, segment):
    '''
    Function that create a binary representation of a segmneted image
    '''
    img = Image.new('RGB', (w, h))

    for x in range(w):
        for y in range(h):
            index = y * w + x
            if segment[index] == 0:
                img.putpixel((x, y), (255, 255, 255))
            else:
                img.putpixel((x, y), (0, 0, 0))


    img.save(path)
    return img

def subsample(img, factor):
    """
    Subsamples an image by averaging the pixels in each block of size `factor x factor`.
    
    Args:
        img PIL object
        factor The subsampling factor
    
    Returns:
        A new PIL Image object representing the subsampled image.
    """
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

def remake_seg_image(original, segmentation, factor, path):
    '''
    Function that turns a binary segmnetation into a segemnated version pof the original
    by combining the two images
    params: img, img, int, string
    '''
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
                    

        if y % factor == (factor - 1):
            segY += 1
    
    final.save(path)
    return final

def accuracy_Jaccard(segImagePath,desiredImagePath):
    '''
    Function that returns the jaccard accuracy of an image
    params: string,string
    '''
    imgSeg = Image.open(segImagePath)
    wS,hS = imgSeg.size
    imgDes = Image.open(desiredImagePath)
    wD,hD = imgDes.size
    totalUnion = 0
    totalInter = 0

    for x in range(wS):
        for y in range(hS):
            if ( imgDes.getpixel((x,y)) == (0,0,0) ) and ( imgSeg.getpixel((x,y)) == (0,0,0) ):
                continue
            elif ( imgDes.getpixel((x,y)) == (0,0,0) ) and ( imgSeg.getpixel((x,y)) != (0,0,0) ):
                totalUnion += 1
            elif ( imgDes.getpixel((x,y)) != (0,0,0) ) and ( imgSeg.getpixel((x,y)) == (0,0,0) ):
                totalUnion += 1
            else:
                totalInter += 1
                totalUnion += 1

    return totalInter/totalUnion

def segment_image(img_path,desired_path, folder="../image_segmentations/"):
    '''
    Function to take an image and perform all three segmentations on it
    Will return timing for each segmentation
    params: string,string,string
    '''
    
    imgOriginal = Image.open(img_path)
    w,h = imgOriginal.size

    factor = w//50 #arbitrary number for subsampling

    #subsampleImage
    subImg = subsample(imgOriginal, factor)

    subWidth, subHeight = subImg.size
    
    pixels = list(subImg.getdata())

    #Setup graphBuilder and the Segmenter objects
    print("Starting...")
    graphBuilder = bd.graphBuilder()
    segmenter = sgm.segment()

    #---------BUILD-GRAPH-&-COMPUTE-EIGENS------------#
    graphBuilder.setup(pixels)
    
    graphBuilder.gaussian_image(subWidth, 10, 0.2, 4, gtype="colour")
    segmenter.setup(graphBuilder.graph)
    print("Graphs + Eigens Done")
    #-------------------------------------------------#

    seg1 = segmenter.ginzburg_landau_segmentation_method(0.01, 21, 0.2, 500) # 50
    img1 = make_seg_img(folder + 'gl_method_binary.jpg', subWidth, subHeight, seg1)


    seg2 = segmenter.fielder_method()
    img2 = make_seg_img(folder + 'fielder_method_binary.jpg', subWidth, subHeight, seg2)

    #segmenter.setup(graphBuilder.graph, laptype="standard")
    seg3 = segmenter.perona_freeman_method(4)
    img3 = make_seg_img(folder + 'perona_freeman_method_binary.jpg', subWidth, subHeight, seg3)
    print("Segmentations Done")

    remake_seg_image(imgOriginal, img1, factor, folder + 'gl_method_combined.jpg')
    remake_seg_image(imgOriginal, img2, factor, folder + 'fielder_method_combined.jpg')
    remake_seg_image(imgOriginal, img3, factor, folder + 'perona_freeman_method_combined.jpg')

    jac1 = accuracy_Jaccard(folder +'fielder_method_combined.jpg', desired_path)
    jac2 = accuracy_Jaccard(folder +'gl_method_combined.jpg', desired_path)
    jac3 = accuracy_Jaccard(folder +'perona_freeman_method_combined.jpg', desired_path)

    print("Fielder Method:")
    print("Jaccard Accuracy  -- ", jac1)
    print()
    print("Perona Freeman Method:")
    print("Jaccard Accuracy  -- ", jac3)
    print()
    print("GL Method:")
    print("Jaccard Accuracy  -- ", jac2)
    print()

def segment_image_tester(img_path, iter,desiredPath, folder="../image_segmentations/"):
    '''
    Function to take an image and perform all three segmentations on it

    Will return timing for each segmentation
    '''
    
    imgOriginal = Image.open(img_path)
    w,h = imgOriginal.size

    factor = w//50 #arbitrary number for subsampling

    #subsampleImage
    subImg = subsample(imgOriginal, factor)

    subWidth, subHeight = subImg.size
    
    pixels = list(subImg.getdata())

    #Setup graphBuilder and the Segmenter objects
    print("Starting...")
    graphBuilder = bd.graphBuilder()
    segmenter = sgm.segment()

    #---------BUILD-GRAPH-&-COMPUTE-EIGENS------------#
    graphBuilder.setup(pixels)

    for i in range(iter):
        x1 = random.random() * 3
        x2 = random.random() * 20
        r = random.randint(5,20)

        print('r=',r, ', sig_g=',x1,', sig_d=',x2)
        graphBuilder.gaussian_image(subWidth, r, x1, x2, gtype="colour")
        segmenter.setup(graphBuilder.graph)
        print("Graphs + Eigens Done")
        #-------------------------------------------------#

        seg1 = segmenter.ginzburg_landau_segmentation_method(0.01, 21, .1, 500)
        img1 = make_seg_img(folder + 'gl_method_binary.jpg', subWidth, subHeight, seg1)

        seg2 = segmenter.fielder_method()
        img2 = make_seg_img(folder + 'fielder_method_binary.jpg', subWidth, subHeight, seg2)

        seg3 = segmenter.perona_freeman_method(8)
        img3 = make_seg_img(folder + 'perona_freeman_method_binary.jpg', subWidth, subHeight, seg3)
        print("Segmentations Done")

        remake_seg_image(imgOriginal, img1, factor, folder + 'gl_method_combined.jpg')
        remake_seg_image(imgOriginal, img2, factor, folder + 'fielder_method_combined.jpg')
        remake_seg_image(imgOriginal, img3, factor, folder + 'perona_freeman_method_combined.jpg')
        print("Images Made Done")
        print(accuracy_Jaccard(folder +'fielder_method_combined.jpg', desiredPath))
        print(accuracy_Jaccard(folder +'gl_method_combined.jpg', desiredPath))
        print(accuracy_Jaccard(folder +'perona_freeman_method_combined.jpg', desiredPath))

def plot_image_eigenfunctions(img_path, folder="../plots/"):
    imgOriginal = Image.open(img_path)
    w,h = imgOriginal.size

    factor = w//50 #arbitrary number for subsampling

    #subsampleImage
    subImg = subsample(imgOriginal, factor)

    subWidth, subHeight = subImg.size
    
    pixels = list(subImg.getdata())

    #Setup graphBuilder and the Segmenter objects
    print("Starting...")
    graphBuilder = bd.graphBuilder()
    segmenter = sgm.segment()

    #---------BUILD-GRAPH-&-COMPUTE-EIGENS------------#
    graphBuilder.setup(pixels)
    
    graphBuilder.gaussian_image(subWidth, 10, 0.2, 4, gtype="colour")
    segmenter.setup(graphBuilder.graph)

    final = Image.new('RGB', (subWidth, subHeight))
    pixelsFinal = list(final.getdata())
    for i in range (0,6):
        eig_2 = np.asarray(segmenter.eigens["vectors"][i])
        maxE = np.max(eig_2)
        
        

        for y in range(subHeight):
            for x in range(subWidth):
                index =  (y*subWidth) + x
                c = int((eig_2[index]/maxE) * 255)
                final.putpixel( (x,y), (c,c,c))



        
        final.save(folder + "imagetester" + str(i) + ".jpg")

