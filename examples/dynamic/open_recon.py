import itertools
import os

import numpy as np
import pyqtgraph as pq
from PyQt5 import QtGui

from examples.dynamic.settings_2020 import SCANS
from fbrct import plot_3d


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
    import vtk

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


reco_dir = "/bigstore/adriaan/recons/evert/2020-02-19 3D paper dataset 1/2020-02-12/pre_proc_20mm_ball_62mmsec_03"

for date, scan_lst in SCANS.items():
    for scan in scan_lst:
        reco_dir = "/bigstore/adriaan/recons/evert/2020-02-19 3D paper dataset 1/"
        reco_dir = os.path.join(reco_dir, date, scan["projs_dir"])

        for t in itertools.count():
            filename = f"{reco_dir}/recon.{t}.vti"
            if os.path.exists(filename):
                continue

            vol_filename = f'{reco_dir}/recon_t{t}.npy'
            if os.path.exists(vol_filename):
                # vol = np.load(f'recon_3d_{fn_pref}_t{t}.npy')[:, :, :]
                vol = np.load(vol_filename)[:, :, :]
                print(f"Converting {vol_filename}...")
                numpy2VTK(vol*200, filename)

                # plot_3d(vol)
                # plot_qtgraph(vol)
                # plot_volume(vol)
                # exit(0)
            else:
                break
