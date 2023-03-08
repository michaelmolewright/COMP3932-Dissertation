from PIL import Image
import networkx as nx
import helper as util
import numpy as np
# Open image using Image module

def messing():

    im = Image.open("./images/megaSmall.jpg")
    pixels = list(im.getdata())
    w, h = im.size
    print(len(pixels))
    G = create_graph(pixels, w, h)
    print("graph made")
    Lnorm = nx.normalized_laplacian_matrix(G)
    print("normalized_laplacian_matrix made")
    eigvalues, eigVectors = np.linalg.eig(Lnorm.A)
    print("eigvalues, eigVectors made")
    eigens = util.sortEigens(eigvalues, eigVectors)
    print("sortEigens done")
    img = Image.new('RGB', (w, h))
    seg = util.second_eigenvector_segmentation(eigens["vectors"][1])
    for x in range(w):
        for y in range(h):
            index = y * w + x
            if seg[index] == 0:
                img.putpixel((x, y), (255, 255, 255))
            else:
                img.putpixel((x, y), (0, 0, 0))
    img.save('output.png')

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


'''
# Get the width and height of the image
width, height = im.size

# Accessing the RGB value of the pixel at coordinates (x, y)
x = 0
y = 0
r, g, b = pixels[y * width + x]

'''