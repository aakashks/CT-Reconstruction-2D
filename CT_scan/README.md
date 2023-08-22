# Imaging Material using CT scan

## Problem
Write a Code in Either MATLAB, C/C++ or Python to solve a general inverse problem. That means N number of detectors are given. Size of Object SOJ is given, Distance between Object to Source and Distance between Source to Detector array (center) is given. Number of Rotations (of source and detector array) is given. AX = B The Code will generate A Matrix by deducing the intercept length under a square Pixel using Simple Geometry and above info. The B matrix is given as well. The code must allow user to give any values of these quantity.


## Mathematical statement
the problem reduces to
$$ A \lambda = d $$
where, $ A $ is the matrix containing information about the intercept of x-rays on the material.
$\lambda$ is the flattened image of the material (attenuation constant data), which will be the final output.
$d$ is made using the given intensity $I_o$ of the  radiation source and 

## References
<https://radiologykey.com/imaging-principles-in-computed-tomography/>
