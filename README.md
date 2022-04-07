# seam-carving
# author: Monee McGrady

The goal of this project was to code up a seam carving algorithm using forward and backward cumulative energy functions presented in the paper
Seam Carving for Content-Aware Image Resizing, as well as to replicate the results presented in the paper, which can be found here: 
http://graphics.cs.cmu.edu/courses/15-463/2012_fall/hw/proj3-seamcarving/imret.pdf.

To run the code, use the following command line template:

python seamcarving.py [source image] [number of seams to add/remove] [shrink/stretch image] [cumulative energy function] [show seams (only for shrinking)]

[source image] = path to the source image you want to shrink or stretch
[number of seams to add/remove] = integer
[shrink/stretch image] = "shrink" or "stretch"
[cumulative energy function] = "forward" or "backward"
[show seams (only for shrinking)] = "yes" or "no"

Examples:
python seamcarving.py input/island.png shrink backward yes

python seamcarving.py input/island.png stretch backward
