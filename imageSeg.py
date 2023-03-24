from PIL import Image
import networkx as nx
import helper as util
import numpy as np
# Open image using Image module

def messing():

    im = Image.open("./images/banana.jpg")
    pixels = list(im.getdata())
    w, h = im.size
    
    print(len(pixels))
    #G = create_graph(pixels, w, h)
    G = create_graph_fully_connected(pixels, w, h, 10, 0.01, 4)
    
    print("graph made")
    Lnorm = nx.normalized_laplacian_matrix(G)
    #affinity = nx.adjacency_matrix(G, weight="weight")
    
    print("normalized_laplacian_matrix made")
    eigvalues, eigVectors = np.linalg.eigh(Lnorm.A)


    
    print("eigvalues, eigVectors made")
    eigens = util.sortEigens(eigvalues, eigVectors)
    
    print("sortEigens done")


    seg1 = util.second_eigenvector_segmentation(eigens["vectors"][1])
    nodes = []
    for i,val in enumerate(pixels):
        nodes.append(i)

    print("second_eigenvector_segmentation done")
    seg2 = util.ginzburg_landau_segmentation(nodes, eigens["values"], eigens["vectors"], 0.1, 1, 2, 500)
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
                    distanceVal = 2.781 ** ( -((pixels_away ** 2)) / sig_x )
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
    img = Image.new('RGB', (w, h))

    for x in range(w):
        for y in range(h):
            index = y * w + x
            if segment[index] == 0:
                img.putpixel((x, y), (255, 255, 255))
            else:
                img.putpixel((x, y), (0, 0, 0))


    img.save(path)
