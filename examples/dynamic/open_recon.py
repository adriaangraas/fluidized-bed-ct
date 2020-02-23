import numpy as np
import vtk
from PyQt5 import QtGui
import pyqtgraph as pq

def plot_qtgraph(vol):
    app = QtGui.QApplication([])
    # This is the way to display a projection angle by angle.
    pq.image(vol)
    pq.image(vol.swapaxes(0, 2))
    app.exec_()


def plot_volume(vol):
    # # postprocess and plot
    # from skimage.restoration import denoise_tv_chambolle
    # y = denoise_tv_chambolle(vol, weight=0.016)
    # plot_3d(y, vmax=0.1)
    y = vol

    from mayavi import mlab
    from skimage.measure import marching_cubes_lewiner

    verts, faces, normals, values = marching_cubes_lewiner(y, 0.15)

    mlab.triangular_mesh([vert[0] for vert in verts],
                         [vert[1] for vert in verts],
                         [vert[2] for vert in verts],
                         faces)
    mlab.show()
    mlab.savefig("")


def numpy2VTK(img, filename, spacing=[1.0, 1.0, 1.0]):
    # evolved from code from Stou S.,
    # on http://www.siafoo.net/snippet/314
    # This function, as the name suggests, converts numpy array to VTK
    importer = vtk.vtkImageImport()

    img_data = img.astype('uint8')
    img_string = img_data.tostring()  # type short
    dim = img.shape

    importer.CopyImportVoidPointer(img_string, len(img_string))
    importer.SetDataScalarType(vtk.VTK_UNSIGNED_CHAR)
    importer.SetNumberOfScalarComponents(1)

    extent = importer.GetDataExtent()
    importer.SetDataExtent(extent[0], extent[0] + dim[2] - 1,
                           extent[2], extent[2] + dim[1] - 1,
                           extent[4], extent[4] + dim[0] - 1)
    importer.SetWholeExtent(extent[0], extent[0] + dim[2] - 1,
                            extent[2], extent[2] + dim[1] - 1,
                            extent[4], extent[4] + dim[0] - 1)

    importer.SetDataSpacing(spacing[0], spacing[1], spacing[2])
    importer.SetDataOrigin(0, 0, 0)

    writer = vtk.vtkXMLImageDataWriter()
    # writer.SetFileDimensionality(3)
    writer.SetInputConnection(importer.GetOutputPort())
    writer.SetFileName(filename)
    writer.UpdateWholeExtent()
    writer.Write()


fn_pref = 'recon_3d_pre_proc_3_30lmin_83mm_FOV2'
# fn_pref = 'recon_pre_proc_6_60lmin_83mm_FOV2'
# fn_pref = 'recon_pre_proc_1_65lmin_83mm_FOV2'
iter = 150

for t in range(100,140):
    vol = np.load(f'{fn_pref}_t{t}_{iter}.npy')[:,:,600:]
    print(f"Converting t={t}...")
    numpy2VTK(vol*200, f"test.{t}.vti")
    # plot_qtgraph(vol)
    # plot_volume(vol)

