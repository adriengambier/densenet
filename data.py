import cv2
from random import randint
import numpy as np

def resize_img(img, scale_min=256, scale_max=480, interpolation=cv2.INTER_NEAREST):
    """
    Resize image with its smaller side sampled in [256, 480]
    :param img: OpenCV image
    :param interpolation: one of the following > cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC
    :return: image resized
    """
    width, height, _ = img.shape
    
    short_side = min(width, height)
    ratio = randint(scale_min, scale_max) / short_side
    
    img_resized = cv2.resize(img,(int(width*ratio),int(height*ratio)), interpolation=interpolation)
    
    return img_resized

def crop_img(img, dim_crop = (224,224)):
    """
    Crop patch from image with dimension dim_crop
    :param img: OpenCV image
    :param dim_crop: dimension of the crop
    :return: image cropped
    """
    width, height, _ = img.shape
    
    crop_x = randint(0, width-dim_crop[0])
    crop_y = randint(0, height-dim_crop[1])
    
    return img[crop_x:crop_x+dim_crop[0], crop_y:crop_y+dim_crop[1]]

# https://gist.github.com/kechan/9a9f4d76f40500b85ce4493e785019ea
def pca_color_augmentation(img):
    """
    Augment image with PCA color augmentation
    :param img: OpenCV image
    :return: image augmented
    """
    
    renorm_image = np.reshape(img, (img.shape[0]*img.shape[1],3))
    
    renorm_image = renorm_image.astype('float32')
    renorm_image -= np.mean(renorm_image, axis=0)
    renorm_image /= np.std(renorm_image, axis=0)

    # Compute covariance matrix
    cov = np.cov(renorm_image, rowvar=False)

    # Compute eigenvalues and eigenvectors of the covariance matrix
    lambdas, p = np.linalg.eig(cov)
    alphas = np.random.normal(0, 0.1, 3)

    # [p1, p2, p3][alpha1*lambda1, alpha2*lambda2, alpha3*lambda3]T
    delta = np.dot(p, alphas*lambdas)

    delta = (delta*255.).astype('int8')

    pca_color_image = np.maximum(np.minimum(img + delta, 255), 0).astype('uint8')
    
    return pca_color_image

def norm_preprocessing(img, mean = None):
    """
    Subtract mean RGB value from each pixel
    :param img: OpenCV image
    :param mean: mean can be provided or otherwise is computed from rgb channels of the image
    :return: normalized image
    """
    
    img = img.astype('float32')
    
    if mean is None:
        img -= np.mean(img, axis=0)
    else:
        img -= mean
    
    return img