"""
name rule:
    VH_XX_AAAAB: XX from top split at AAAA,
     B: 1 - stands for origin split, 2 - flip, 3 - rotation,
usage:
    1) setting the folder as indicated structure (creating the split folder to contain the augmented images)
    2) run the x_VH_split() wiht selected patch size
    3) generate the file index is optional
"""
import numpy as np
from PIL import Image
import os
import cv2
# Configure here to set the data config files


training_set = [1, 3, 5, 7, 13, 17, 21, 23, 26, 32, 37] # VH training index
#test_set = [11, 15, 28, 30, 34] # VH test index
test_set = [15]

def x_calc_lucorners(im, patch_size, overlap_size=[0, 0]):
    '''
    :param im:
    :param patch_size:
    :param overlap_size: [0, Patchsize[*]] indicating how much is the overlaping
    :param boarderflag: is the flag marking the way to deal with the boarder
        - set 1(default): to discard the residual pixels
        - set 2 : to reflect along the boarder
    :return:
        corners: a list including the corner coordinates [x,y]
    '''
    im_arr = np.array(im)
    # shape is always height first
    (im_height, im_width) = (im_arr.shape[0], im_arr.shape[1])
    l_x = []
    l_y = []
    cur_x = 0
    cur_y = 0
    while cur_x  < im_height:
        while cur_y < im_width:
            if (cur_x + patch_size[0] <= im_height) and (cur_y + patch_size[1] <= im_width):
                l_x.append(cur_x)
                l_y.append(cur_y)
            elif (cur_x + patch_size[0] > im_height) and (cur_y + patch_size[1] <= im_width):
                l_x.append(im_height - patch_size[0])
                l_y.append(cur_y)
            elif (cur_x + patch_size[0] <= im_height) and (cur_y + patch_size[1] > im_width):
                l_x.append(cur_x)
                l_y.append(im_width - patch_size[1])
            else:
                l_x.append(im_height - patch_size[0])
                l_y.append(im_width - patch_size[1])

            cur_y = cur_y + patch_size[1] - overlap_size[1]

        cur_x = cur_x + patch_size[0] - overlap_size[0]
        cur_y = 0
    return [l_x, l_y]

def x_calc_lucorners1(im, patch_size, overlap_size=[0, 0]):
    '''
    :param im:
    :param patch_size:
    :param overlap_size: [0, Patchsize[*]] indicating how much is the overlaping
    :param boarderflag: is the flag marking the way to deal with the boarder
        - set 1(default): to discard the residual pixels
        - set 2 : to reflect along the boarder
    :return:
        corners: a list including the corner coordinates [x,y]
    '''
    im_arr = np.array(im)
    # shape is always height first
    (im_height, im_width) = (im_arr.shape[0], im_arr.shape[1])
    l_x = []
    l_y = []
    cur_x = 0
    cur_y = 0
    while cur_x  < im_height:
        while cur_y < im_width:
            l_x.append(cur_x)
            l_y.append(cur_y)

            cur_y = cur_y + patch_size[1] - overlap_size[1]

        cur_x = cur_x + patch_size[0] - overlap_size[0]
        cur_y = 0
    return [l_x, l_y]

def x_one_split(l_cor, patch_size, origin_image):
    '''
    split one image PATCH at l_cor position
    :param l_cor: the corner list, indicating the up left corner
    :param patch_size:
    :param origin_image:
    :return:
     split image
    '''
    box = (l_cor[1], l_cor[0], l_cor[1]+patch_size[1], l_cor[0]+patch_size[0])
    c_im = origin_image.crop(box)
    return c_im




def x_image_split(l_corners, patch_size, origin_image, root_name):
    '''
    Split one image PATCH with augmentation: flip H, flip V, rotation -> x8
    :param l_corners: corner list
    :param patch_size:
    :param origin_image: input image tile
    :param root_name: save root folder name
    :return:
    '''
    n_corners = len(l_corners[0])
    for i in range(n_corners):
        fname = root_name + str(i).zfill(4) + str(0) + '.png'
        l_cor = [l_corners[0][i], l_corners[1][i]]
        c_im = x_one_split(l_cor, patch_size, origin_image)
        c_im.save(fname)
        # data augmentation
        # 1. filp lr
        fname = root_name + str(i).zfill(4) + str(1) + '.png'
        tc_im = c_im.transpose(Image.FLIP_LEFT_RIGHT)
        tc_im.save(fname)
        # 2. flip ud + r90
        fname = root_name + str(i).zfill(4) + str(2) + '.png'
        tc_im = c_im.transpose(Image.FLIP_TOP_BOTTOM)
        tc_im.save(fname)
        fname = root_name + str(i).zfill(4) + str(3) + '.png'
        tc_im = tc_im.rotate(90)
        tc_im.save(fname)
        fname = root_name + str(i).zfill(4) + str(4) + '.png'
        tc_im = tc_im.rotate(180)
        tc_im.save(fname)
        # 4. rotation 90 180 270
        cdx = 5
        for ra in range(90, 360, 90):
            fname = root_name + str(i).zfill(4) + str(cdx) + '.png'
            tc_im = c_im.rotate(ra)
            tc_im.save(fname)
            cdx = cdx + 1


def x_image_split_no_aug(l_corners, patch_size, origin_image, root_name):
    '''
    split the origin_image into patch size, with the direction given by l_corners
    WITHOUT augmentation -> for test dataset preprocessing
    :param l_corners: a list containing the spliting strating points
    :param patch_size: the patch size
    :param origin_image: original image which is the tile
    :param root_name: saving folder and prefix
    :return:
    '''
    n_corners = len(l_corners[0])
    for i in range(n_corners):
        fname = root_name + str(i).zfill(4) + str(0) + '.png'
        l_cor = [l_corners[0][i], l_corners[1][i]]
        c_im = x_one_split(l_cor, patch_size, origin_image)
        c_im.save(fname)

# -------------------------------------------------------------------- #
#                  Function to call directly                           #
# -------------------------------------------------------------------- #
def xf_VH_train_split(ps, overlap):
    '''
    Split the VH dataset with spcified patch_size;
    spliting both RGB data and the TAG data
    dave to
    t_rgb_folder and t_tag_folder
    modify the string to save it to different place
    :param ps: a tuple including the [height, width] of the patch
    :param overlap: Define the overlap size [height, width]
    :return:
    '''
    # modify here change the overlap coefficient
    patch_size = ps
    overlap_size = [int(patch_size[0]*overlap[0]), int(patch_size[1]*overlap[1])]


    for idx in training_set:
        # split and augmentation of the RGB data

        folder_root = "/home/f523/wangyang/segmentation/Vaihingen/ISPRS_semantic_labeling_Vaihingen/top/"
        t_rgb_folder = "/home/f523/wangyang/segmentation/Vaihingen/Split/original/"
        fname = folder_root + 'top_mosaic_09cm_area' + str(idx) + '.tif'
        im = Image.open(fname)
        l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
        root_name = t_rgb_folder + 'VH_' + str(idx).zfill(2) + '_'
        x_image_split(l_corners, patch_size, im, root_name)

        # split and augmentation of the Tag data   /home/f523/wangyang/segmentation/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE
        folder_root = "/home/f523/wangyang/segmentation/Vaihingen/label_train/"
        t_tag_folder = "/home/f523/wangyang/segmentation/Vaihingen/Split/label_train/"
        fname = folder_root + 'top_mosaic_09cm_area' + str(idx) + '.png'
        im = Image.open(fname,'r')
        l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
        root_name = t_tag_folder + 'VH_' + str(idx).zfill(2) + '_'
        x_image_split(l_corners, patch_size, im, root_name)

        folder_root = "/home/f523/wangyang/segmentation/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/"
        t_tag_folder = "/home/f523/wangyang/segmentation/Vaihingen/Split/label/"
        fname = folder_root + 'top_mosaic_09cm_area' + str(idx) + '.tif'
        im = Image.open(fname,'r')
        l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
        root_name = t_tag_folder + 'VH_' + str(idx).zfill(2) + '_'
        x_image_split(l_corners, patch_size, im, root_name)
        # split and augmentation of the no-boarder tag data

def xf_VH_test_split(ps, overlap):
    '''
    Split the VH dataset with spcified patch_size;
    spliting both RGB data and the TAG data
    dave to
    t_rgb_folder and t_tag_folder
    modify the string to save it to different place
    :param ps: a tuple including the [height, width] of the patch
    :param overlap: Define the overlap size [height, width]
    :return:
    '''
    # modify here change the overlap coefficient
    patch_size = ps
    overlap_size = [int(patch_size[0]*overlap[0]), int(patch_size[1]*overlap[1])]


    for idx in test_set:
        # split and augmentation of the RGB data
        folder_root = "/home/wangyang/Desktop/dataset/Vaihingen/ISPRS_semantic_labeling_Vaihingen/top/"
        t_rgb_folder = "/home/wangyang/Desktop/dataset/Vaihingen/Split/test/15/original/256/"
        fname = folder_root + 'top_mosaic_09cm_area' + str(idx) + '.tif'
        im = Image.open(fname)
        l_corners = x_calc_lucorners1(im, patch_size, overlap_size=overlap_size)
        root_name = t_rgb_folder + 'VH_' + str(idx).zfill(2) + '_'
        x_image_split_no_aug(l_corners, patch_size, im, root_name)

        folder_root = "/home/wangyang/Desktop/dataset/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/"
        t_tag_folder = "/home/wangyang/Desktop/dataset/Vaihingen/Split/test/15/label/256/"
        fname = folder_root + 'top_mosaic_09cm_area' + str(idx) + '.tif'
        im = Image.open(fname)
        l_corners = x_calc_lucorners1(im, patch_size, overlap_size=overlap_size)
        root_name = t_tag_folder + 'VH_' + str(idx).zfill(2) + '_'
        x_image_split_no_aug(l_corners, patch_size, im, root_name)

        folder_root = "/home/wangyang/Desktop/dataset/Vaihingen/ISPRS_semantic_labeling_Vaihingen/ndsm/"
        t_tag_folder = "/home/wangyang/Desktop/dataset/Vaihingen/Split/test/15/ndsm/256/"
        fname = folder_root + 'dsm_09cm_matching_area' + str(idx) + '_normalized.jpg'#dsm_09cm_matching_area3_normalized
        im = Image.open(fname)
        l_corners = x_calc_lucorners1(im, patch_size, overlap_size=overlap_size)
        root_name = t_tag_folder + 'VH_' + str(idx).zfill(2) + '_'
        x_image_split_no_aug(l_corners, patch_size, im, root_name)

        folder_root = "/home/wangyang/Desktop/dataset/Vaihingen/label_train/"
        t_tag_folder = "/home/wangyang/Desktop/dataset/Vaihingen/Split/test/15/tag/256/"
        fname = folder_root + 'top_mosaic_09cm_area' + str(idx) + '.png'#dsm_09cm_matching_area3_normalized
        im = Image.open(fname)
        l_corners = x_calc_lucorners1(im, patch_size, overlap_size=overlap_size)
        root_name = t_tag_folder + 'VH_' + str(idx).zfill(2) + '_'
        x_image_split_no_aug(l_corners, patch_size, im, root_name)

def xh_test():
    # define the roots
    patch_size = [256, 256]
    overlap = [0, 0]#0.297
    # xf_VH_test_split(patch_size, overlap)
    xf_VH_test_split(patch_size, overlap)


if __name__ == '__main__':
    # x_write_to_index(' ', './train_complete.txt')
    xh_test()
