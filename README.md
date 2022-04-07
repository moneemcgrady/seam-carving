# seam-carving
# author: Monee McGrady

The goal of this project was to code up a seam carving algorithm using forward and backward cumulative energy functions presented in the paper
Seam Carving for Content-Aware Image Resizing, which can be found here: <br />
http://graphics.cs.cmu.edu/courses/15-463/2012_fall/hw/proj3-seamcarving/imret.pdf. <br />

To run the code, use the following command line template: <br />

python seamcarving.py [source image] [number of seams to add/remove] [shrink/stretch image] [cumulative energy function] [output image path] [show seams (only for shrinking)] <br />

[source image] = path to the source image you want to shrink or stretch <br />
[number of seams to add/remove] = integer <br />
[shrink/stretch image] = "shrink" or "stretch" <br />
[cumulative energy function] = "forward" or "backward" <br />
[output image path] = path to output image <br />
[show seams (only for shrinking)] = "yes" or "no" <br />

Examples: <br />
python seamcarving.py input/island.png 50 shrink backward output/island_red_lines.png yes <br />

python seamcarving.py input/island.png 50 stretch output/island_stretch.png backward <br />
