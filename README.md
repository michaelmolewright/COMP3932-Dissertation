# COMP3932 Synoptic Project

## Name
An implementation of 3 graph segmentation algorithms in Python

## Description
This project consists of a class that can build graphs from spatial data sets and images using different weight functions. There is also a class that can then segment these graphs using 3 seperate segmentation algorithms. All of the algorithms use the graph Laplacian. A PDF is also included which is my final report. Paramters can be changed in the code if needed and the classes can be used more generally if needed. Please note, the helper.py and misc.py contain functions that may have been useful, but arent used in the final solution

To run the code simply type "python3 main.py" whilst in the code directory.

Here is a directory chart:
```
.
├── Code/                   # Contains all of the Python Code
├── Image_segmenations/     # Folder for image segmenations
├── Images/                 # Folder for Stock images
├── plots/                  # All of the Two Moons plots and other graphs generated
├── requirements.txt        # Contains all of the appropriate requirements
├── .gitignore              # file that holds infomation about files to ignore
├── README.md 
└── WRIGHT23-FINAL.pdf      # Final report
```
## Installation
A requirements.txt file is included in this repository. To install and run the code, open a virtual environment in Python and type the command "pip install -r requirements.txt". A virtual environment isnt necassary however it could make development more streamlined.

## Authors and acknowledgment
Michael Wright