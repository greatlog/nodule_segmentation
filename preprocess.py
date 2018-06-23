from multiprocessing import Pool, cpu_count,freeze_support
import numpy as np
import SimpleITK as sitk
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.measure import label, regionprops
from skimage.filters import roberts
from skimage import measure
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
from scipy.ndimage.interpolation import zoom
import vtk  # sudo apt-get install python-vtk
from vtk.util import numpy_support
import warnings
import matplotlib.pyplot as plt


def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing


def load_dicom(dir_path):
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(dir_path)
    reader.Update()
    spacing = list(reader.GetPixelSpacing())  # x, y, z
    spacing.reverse()  # z, y, x
    _extent = reader.GetDataExtent()
    ConstPixelDims = [
        _extent[1] - _extent[0] + 1, _extent[3] - _extent[2] + 1,
        _extent[5] - _extent[4] + 1
    ]
    imageData = reader.GetOutput()
    pointData = imageData.GetPointData()
    assert (pointData.GetNumberOfArrays() == 1)
    arrayData = pointData.GetArray(0)
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')
    ArrayDicom = np.rot90(ArrayDicom, 1)
    ArrayDicom = np.transpose(ArrayDicom, [2, 0, 1])
    return np.array(spacing), ArrayDicom


def ReadDICOMFolder(folderName, input_uid=None):
    '''A perfect DCM reader!'''
    reader = sitk.ImageSeriesReader()
    out_uid = ''
    out_image = None
    max_slice_num = 0
    # if the uid is not given, iterate all the available uids
    uids = [input_uid] if input_uid != None else reader.GetGDCMSeriesIDs(folderName)
    for uid in uids:
        try:
            dicomfilenames = reader.GetGDCMSeriesFileNames(folderName, uid)
            reader.SetFileNames(dicomfilenames)
            image = reader.Execute()
            size = image.GetSize()
            if size[0] == size[1] and size[2] != 1:  # exclude xray
                slice_num = size[2]
                if slice_num > max_slice_num:
                    out_image = image
                    out_uid = uid
                    max_slice_num = slice_num
        except:
            pass
    if out_image != None:
        imageRaw = sitk.GetArrayFromImage(out_image)
        imageRaw = np.flip(imageRaw, 0)
        spacing = out_image.GetSpacing()
        return imageRaw, spacing
    else:
        raise Exception('Fail to load the dcm folder.')


def get_segmented_lungs_mask(im, i, plot=False):
    size = im.shape[1]
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < -320
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    temp_label = label(cleared)
    for region in regionprops(temp_label):
        if region.area < 50:
            # print region.label
            for coordinates in region.coords:
                temp_label[coordinates[0], coordinates[1]] = 0
    cleared = temp_label > 0
    cleared = ndi.binary_dilation(cleared, iterations=5)
    ###################################################
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    for region in regionprops(label_image):
        if region.eccentricity > 0.99 \
                or region.centroid[0] > 0.90 * size \
                or region.centroid[0] < 0.12 * size \
                or region.centroid[1] > 0.88 * size \
                or region.centroid[1] < 0.10 * size \
                or (region.centroid[1] > 0.46 * size and region.centroid[1] < 0.54 * size and region.centroid[
            0] > 0.75 * size) \
                or (region.centroid[0] < 0.2 * size and region.centroid[1] < 0.2 * size) \
                or (region.centroid[0] < 0.2 * size and region.centroid[1] > 0.8 * size) \
                or (region.centroid[0] > 0.8 * size and region.centroid[1] < 0.2 * size) \
                or (region.centroid[0] > 0.8 * size and region.centroid[1] > 0.8 * size):
            for coordinates in region.coords:
                label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    # print "step 7", time()-t1
    return binary, i


def mask_extractor(path, area_th=5e6, min_area=1000, debug=False):
    '''Get the 3d mask of a CT with multiprocessing'''

    try:
        imgs, spc= ReadDICOMFolder(path)
    except:
        try:
            print('using vtk to load the dcm ... ')
            spc, imgs= load_dicom(path)
        except:
            print('bad case:', path)

    assert spc[0] < 5, "slice thickness must <5mm!"

    # binarize each frame with multiprocesses
    pool = Pool(cpu_count())
    results = []
    for i in range(imgs.shape[0]):
        result = pool.apply_async(get_segmented_lungs_mask, (
            imgs[i],
            i,
        ))
        results.append(result)
    pool.close()
    pool.join()
    # res is an ApplyResult Object, use .get() to obtain data
    results = [res.get() for res in results]
    im_mask = np.zeros_like(imgs, dtype=np.bool)
    for (msk, i) in results:
        im_mask[i] = msk

    # label every 3d-connected region
    label = measure.label(im_mask)
    properties = measure.regionprops(label)
    assert len(properties) > 0, "empty image"
    properties.sort(key=lambda x: x.area, reverse=True)
    # keep the largest connected region
    # keep any region that is larger than area_th
    valid_label = set()
    for index, prop in enumerate(properties):
        if index == 0:
            valid_label.add(prop.label)
        else:
            if prop.area > area_th:
                valid_label.add(prop.label)
            else:
                break
    # np.in1d: return a list of bools with the same length as 1d label. True if it is in valid_label.
    current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
    # delete a few starting and ending frames
    for i in range(current_bw.shape[0]):
        if current_bw[i].sum() < min_area:
            current_bw[i] = 0
        else:
            break
    for i in reversed(range(current_bw.shape[0])):
        if current_bw[i].sum() < min_area:
            current_bw[i] = 0
        else:
            break

    if debug:
        for i in range(current_bw.shape[0]):
            current_bw[i].sum()
    # 3d-dilation for making sure that marginal nodules are involved
    mask_2 = ndi.binary_dilation(current_bw, iterations=25)
    if debug:
        plt.imshow(mask_2[mask_2.shape[0] / 2], cmap="bone")
    return imgs, mask_2, spc


def lumTrans(img, lungwin=[-1000., 600.]):
    '''Window level and normalization'''
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


def resample(imgs, spacing, new_spacing, order=2):
    '''Because of rounding operation, the true spacing might be different
    from the new_spacing.
    If the input is a 4d data, squeeze to 3d and apply this function.
    '''
    if len(imgs.shape) == 3:
        new_shape = np.round(np.array(imgs.shape) * np.array(spacing) / np.array(new_spacing))
        true_spacing = np.array(spacing) * np.array(imgs.shape) / np.array(new_shape)
        resize_factor = np.array(new_shape) / np.array(imgs.shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
        return imgs, true_spacing
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:, :, :, i]
            newslice, true_spacing = resample(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError('wrong shape')


def mask_processing(im, mask, spacing):
    ''''''
    resolution = np.array([1, 1, 1])
    # mm shape of the iamge & mask
    newshape = np.round(np.array(mask.shape) * spacing / resolution)
    xx, yy, zz = np.where(mask)
    # 3d bounding box of the lung area, shape = (2,3)
    box = np.array([[np.min(xx), np.max(xx)],
                    [np.min(yy), np.max(yy)],
                    [np.min(zz), np.max(zz)]])
    # transform the scale from voxel to mm
    box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
    box = np.floor(box).astype('int')
    margin = 0
    # extend the bounding box by margin with the constraint of image size
    extendbox = np.vstack([
        np.max([[0, 0, 0], box[:, 0] - margin], 0),
        np.min([newshape, box[:, 1] + 2 * margin], axis=0).T
    ]).T
    extendbox = extendbox.astype('int')

    bone_thresh = 210
    pad_value = 170

    sliceim = lumTrans(im)
    sliceim = sliceim * mask + pad_value * (1 - mask).astype('uint8')
    bones = sliceim * mask > bone_thresh
    sliceim[bones] = pad_value

    # sliceim1, true_spacing = resample(sliceim, spacing, resolution, order=1)
    # sliceim2 = sliceim1[extendbox[0, 0]:extendbox[0, 1], extendbox[1, 0]:
    #                                                      extendbox[1, 1], extendbox[2, 0]:extendbox[2, 1]]
    #
    # sliceim = sliceim2[np.newaxis, ...]
    return sliceim, extendbox

