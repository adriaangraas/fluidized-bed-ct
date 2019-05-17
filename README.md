# Fluidized bed CT 
Two **real-data examples**:
- reconstruction on a plane in the volume using a [fanbeam approximation](examples/fanbeam_approx.ipynb);
- partial 3D reconstruction using [conebeam](examples/conebeam_3d.ipynb).

And, to get to understand the qualities of tomographic reconstruction, some experiments on 
**numerical phantoms**:
- study of the artifacts produced by using a fanbeam approximation in a [2D slice](examples/phantom_2d.ipynb);
- study of the SART reconstruction method and median filter working on a [3D ellipsoid](examples/phantom_3d.ipynb) that travels through the column.
- study of reconstructions when [two bubbles](examples/double_bubble_2d.ipynb) go through a slice 
and overlap on a detector.

![](bubble.png) 

### Note on runtime performance 
This code runs mostly on the CPU, and some important iterations are done in 
Python: it is not written with performance in mind. It can be made perhaps a factor 100 or so 
more performant with a few adjustments, some of which are quite simple.

### Installation instructions
You need to install ODL, ASTRA Toolbox as well as scipy and numpy.

### License
This project is licensed under the GNU General Public License v3 - see the [LICENSE](LICENSE) file for details.
