import numpy as np
from PIL import Image
import os
import cv2


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
    while cur_x < im_height:
        while cur_y< im_width:
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
    #box = (l_cor[1], l_cor[0], l_cor[1]+patch_size[1], l_cor[0]+patch_size[0])
    c_im = origin_image[l_cor[0]:l_cor[0]+patch_size[0],l_cor[1]:l_cor[1]+patch_size[1]]
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
        cv2.imwrite(fname,c_im)
        # data augmentation
        # 1. filp lr
        fname = root_name + str(i).zfill(4) + str(1) + '.png'
        tc_im = cv2.flip(c_im,1,dst=None)
        #tc_im = c_im.transpose(Image.FLIP_LEFT_RIGHT)
        cv2.imwrite(fname,tc_im)

        # 2. flip ud + r90
        fname = root_name + str(i).zfill(4) + str(2) + '.png'
        tc_im = cv2.flip(c_im,0,dst=None)
        #tc_im = c_im.transpose(Image.FLIP_TOP_BOTTOM)
        cv2.imwrite(fname,tc_im)

        fname = root_name + str(i).zfill(4) + str(3) + '.png'
        tc_im = cv2.rotate(tc_im,0)
        #tc_im = tc_im.rotate(90)
        cv2.imwrite(fname,tc_im)
        fname = root_name + str(i).zfill(4) + str(4) + '.png'
        tc_im = cv2.rotate(tc_im,1)
        #tc_im = tc_im.rotate(180)
        cv2.imwrite(fname,tc_im)
        # 4. rotation 90 180 270
        cdx = 5
        for ra in range(0, 1, 2):
            fname = root_name + str(i).zfill(4) + str(cdx) + '.png'
            tc_im = cv2.rotate(c_im,ra)
            cv2.imwrite(fname,tc_im)
            cdx = cdx + 1
def xf_PD_data_split_train(patch_size, overlap):
    """
    Split the Potsdam dataset with certain overlap and patch size
    :param overlap: [height, weight] in scale [0,1]
    :param patch_size: [height, weight] in pixels
    :return:
    """
    print ('spliting the training data')

    tag_folder = '/home/wangyang/Desktop/dataset/potsdam/train_label/'


    n_tile1 = [2, 3, 4, 5, 6, 7]
    n_tile2 = [7, 8, 9, 10, 11]
    #n_tile1 = [2,3,4,5,6,7]
    #n_tile2 = [12]
    prefix = 'top_potsdam_'
    sufix = 'RGB.tif'
    data_path = "/home/wangyang/Desktop/dataset/potsdam/2_Ortho_RGB/"
    tar_root = "/home/wangyang/Desktop/dataset/potsdam/Split/"
    #tar_root = '../PD/split/train/'

    overlap_size = [int(patch_size[0] * overlap[0]), int(patch_size[1] * overlap[1])]
    fid = 0
    for i1 in n_tile1:
        #print("in")
        for i2 in n_tile2:
            src_fname = data_path +'top_potsdam_' + str(i1) + '_' + str(i2) + '_' + sufix
            if os.path.isfile(src_fname):
                tar_path = tar_root + 'original/'
                im = cv2.imread(src_fname)
                print(np.array(im).shape)
                l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
                root_name = tar_path + 'PD_' + str(fid).zfill(2) + '_'
                x_image_split(l_corners, patch_size, im, root_name)


                src_fname = tag_folder + 'top_potsdam_' + str(i1) + '_' + str(i2) + '_label.png'
                tar_path = tar_root + 'label_train/'
                im = cv2.imread(src_fname)
                l_corners = x_calc_lucorners(im, patch_size, overlap_size=overlap_size)
                root_name = tar_path + 'PD_' + str(fid).zfill(2) + '_'
                x_image_split(l_corners, patch_size, im, root_name)


                #x_image_split_no_aug(l_corners, patch_size, im, root_name)
                fid += 1
def pd_test():
    # define the roots
    patch_size = [336, 336]
    overlap = [0.19, 0.19]
    # xf_VH_test_split(patch_size, overlap)
    xf_PD_data_split_train(patch_size, overlap)


if __name__ == '__main__':
    # x_write_to_index(' ', './train_complete.txt')
    pd_test()
