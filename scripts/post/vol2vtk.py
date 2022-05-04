import itertools

from settings import *


def numpy2VTK(img, filename, spacing=(1., 1., 1.)):
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


def load_and_convert(in_filename, out_filename):
    vol = np.load(in_filename)[:, :, :]
    print(f"Converting {in_filename}...")
    numpy2VTK(vol * 200, out_filename)


# RECO_DIR = "/bigstore/adriaan/recons/evert/hr-corrected-2020-2/"
RECO_DIR = "/home/adriaan/data/evert/recons/"

for scan in SCANS:
    reco_dir = os.path.join(RECO_DIR, scan.name,
                            "size_200_algo_sirt_iters_350")

    if not os.path.exists(reco_dir):
        print(reco_dir + " does not exist.")
        continue

    vol_filenames = {}
    for t in itertools.count():
        vol_filename = f'{reco_dir}/recon_t{str(t).zfill(6)}.npy'

        if not os.path.exists(vol_filename):
            print("No more volumes found, breaking.")
            break

        vol_filenames[f"{reco_dir}/recon.{t}.vti"] = vol_filename

        # vol_filename = f'{reco_dir}/recon_t{str(0).zfill(6)}.npy'
        # vol_filenames[f"{reco_dir}/recon.{0}.vti"] = vol_filename
        # vol_filenames[f'{reco_dir}/recon_median.vti'] = f'{reco_dir}/recon_from_median.npy'

    for filename, vol_filename in vol_filenames.items():
        # if not os.path.exists(filename):
        load_and_convert(vol_filename, filename)
