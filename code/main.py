
import twoMoonsSeg as tm
import imageSeg as mk


if __name__ == "__main__":
    '''
    Area to run code from the other python files
    '''

    print("Running the Two Moons Test")
    #  Runs a segmentation of the two moons data set
    #  All of the segmentations will be performed, parameters can be changed within the function
    tm.experiment(1000,0.15, 1)


    print("Running Segmentation of an Image")
    #  Runs a segmentation of the file path, and computes an accuracy based on the desired image file path
    #  All of the segmentations will be performed, parameters can be changed within the function
    mk.segment_image('../images/plane.jpg','../images/planeDesired.jpg')