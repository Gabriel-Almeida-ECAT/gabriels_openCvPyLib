import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import copy

from . import bezier

matplotlib.use('TkAgg')


def mapValue(var, var_min, var_max, ret_min, ret_max):
    if var < var_min or var > var_max:
        exit(1)
    return (ret_max - ret_min)*((var - var_min)/(var_max - var_min))+ret_min


# credit to 'Alexey Antonenko' in https://stackoverflow.com/questions/23660929/how-to-check-whether-a-jpeg-image-is-color-or-gray-scale-using-only-python-stdli
def isGray(img_obj):
    if len(img_obj.shape) < 3: return True
    if img_obj.shape[2]  == 1: return True
    b,g,r = img_obj[:,:,0], img_obj[:,:,1], img_obj[:,:,2]
    if (b==g).all() and (b==r).all(): return True
    return False


#implementar teste para outros formats (SVG, HSV)
def showImages(imgs_list, titles_list=None, **kwargs):
    #plt.figure(figsize=(10, 5))
    num_cols = kwargs.get('num_cols')
    num_rows = kwargs.get('num_rows')
    if num_cols is None and num_rows is None:
        num_rows = math.ceil(len(imgs_list)/3)
        num_cols = len(imgs_list) if len(imgs_list) < 3 else 3
    elif num_rows is None: #user specified number of cols
        num_rows = math.ceil(len(imgs_list)/num_cols)
    elif num_cols is None: #user specified number of cols
        num_cols = math.ceil(len(imgs_list)/num_rows)

    for ind, img in enumerate(imgs_list, start=1):
        plt.subplot(num_rows, num_cols, ind)
        if isGray(img):
            plt.imshow(img, cmap='gray')
        else:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(rgb_img)
        if titles_list != None:
            plt.title(titles_list[ind-1])
        if not kwargs.get('axis', False):
          plt.axis('off')

    plt.tight_layout()
    plt.show()


def addImagesProp(img_obj1, img_obj2, **kwargs):
    if kwargs.get('w1') is None and kwargs.get('w1') is None:
        w1, w2 = 0.5, 0.5
    elif kwargs.get('w1') is None:
        w1 = 1 - kwargs.get('w2')
    elif kwargs.get('w2') is None:
        w2 = 1 - kwargs.get('w1')

    return cv2.addWeighted(img_obj1, w1, img_obj2, w2, 0)


def bgr2gray(img_obj):
    return cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)


def bgr2rgb(img_obj):
    return cv2.cvtColor(img_obj, cv2.COLOR_BGR2RGB)


def invertImg(img_obj):
    return cv2.bitwise_not(img_obj)


def setHSValues(img_obj, **kwargs):
    hsv_img = cv2.cvtColor(img_obj, cv2.COLOR_BGR2HSV)
    (h, s, v) = cv2.split(hsv_img)
    h = np.uint8(np.clip(s*kwargs.get('hue', 1), 0,255))
    s = np.uint8(np.clip(h*kwargs.get('saturation', 1), 0,255))
    v = np.uint8(np.clip(v*kwargs.get('value', 1), 0,255))
    return cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2BGR)


def zoomImg(img_obj, interval_x, interval_y):
    zoomed_img = img_obj[interval_x[0] : interval_x[1] , interval_y[0] : interval_y[1]]
    #imshow(zoomed_img)
    return zoomed_img


def rgb2yiq(rgb):
    y = rgb @ np.array([[0.30], [0.59], [0.11]])

    rby = rgb[:, :, (0, 2)] - y # In this case, (0, 2) indicates that you want to select the first and third channels (0-indexed) from the last dimension of the rgb array.
    i = np.sum(rby * np.array([[[0.74, -0.27]]]), axis=-1)
    q = np.sum(rby * np.array([[[0.48, 0.41]]]), axis=-1)

    yiq = np.dstack((y.squeeze(), i, q))
    return yiq


def bgr2yiq(bgr):
    rgb = np.float32(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    yiq = rgb2yiq(rgb)

    return yiq


def yiq2rgb(yiq):
    r = yiq @ np.array([1.0, 0.9468822170900693, 0.6235565819861433])
    g = yiq @ np.array([1.0, -0.27478764629897834, -0.6356910791873801])
    b = yiq @ np.array([1.0, -1.1085450346420322, 1.7090069284064666])
    rgb = np.clip(np.dstack((r, g, b)), 0, 255)
    return np.uint8(rgb)


def averageBlur(img_obj, **kwargs):
    kernel_size = kwargs.get("kernel_size", (5,5))
    return cv2.blur(img_obj, kernel_size)


# 'median'[eng] =/= 'media'[pot] !!! toda vez eu esqueço -_-
def medianBlur(img_obj, **kwargs):
    kernel_size = kwargs.get("kernel_size", 5)
    return cv2.medianBlur(img_obj, kernel_size)


# just to have a example
def meadianBlurConv(img_obj, **kwargs):
    kernel_size = kwargs.get("kernel_size", 5)
    kernel = np.ones((kernel_size, kernel_size), np.float32)/kernel_size**2 #median filter 5x5
    return cv2.filter2D(img_obj, -1, kernel)


def gaussianBlur(img_obj, **kwargs):
    kernel_size = kwargs.get("kernel_size", (5,5))
    return cv2.GaussianBlur(img_obj, kernel_size, 0)


# good ideia to reduce noise (gaussian filter) before usage
def sobelFilter(img_obj, **kwargs):
    kernel_size = kwargs.get("kernel_size", 3)
    scale = kwargs.get("scale", 1)
    delta = kwargs.get("delta", 0)
    ddepth = kwargs.get("ddepth", cv2.CV_16S) #ou cv2.CV_8U

    gray_img = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray_img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray_img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    return addImagesProp(abs_grad_x, abs_grad_y) #see if theres any diference with 'bitwise_or' func


# good ideia to reduce noise (gaussian filter) before usage
def robertsFilter(img_obj, **kwargs):
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])

    gray_img = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)

    img_robert_x = cv2.filter2D(gray_img, -1, kernel_x)
    img_robert_y = cv2.filter2D(gray_img, -1, kernel_y)

    return addImagesProp(img_robert_x, img_robert_y)

# good ideia to reduce noise (gaussian filter) before usage
def prewittFilter(img_obj, **kwargs):
    kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernel_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    gray_img = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)

    img_robert_x = cv2.filter2D(gray_img, -1, kernel_x)
    img_robert_y = cv2.filter2D(gray_img, -1, kernel_y)

    return addImagesProp(img_robert_x, img_robert_y)


# good ideia to reduce noise (gaussian filter) before usage
def laplacianFilter(img_obj, **kwargs):
    kernel_size = kwargs.get("kernel_size", 3)
    ddepth = kwargs.get("ddepth", cv2.CV_16S)  # ou cv2.CV_8U

    gray_img = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)

    lap_filt_img = cv2.Laplacian(gray_img, ddepth, ksize=kernel_size)

    return cv2.convertScaleAbs(lap_filt_img)


# good ideia to reduce noise (gaussian filter) before usage
def cannyFilter(img_obj, **kwargs):
    min_val = kwargs.get("min_val", 100)
    max_val = kwargs.get("max_val", 200)
    aperture_size = kwargs.get("aperture_size", 3)

    gray_img = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)
    canny_filt_img = cv2.Canny(gray_img, min_val, max_val, aperture_size, L2gradient=True)

    return cv2.convertScaleAbs(canny_filt_img)


def plainAbsoluteThresholding(img_obj, **kwargs):
    thr_val = kwargs.get("thr_val", 122)
    max_val = kwargs.get("max_val", 255)

    ret, bin_img = cv2.threshold(getGrayImg(img_obj), thr_val, max_val, cv2.THRESH_BINARY)

    return ret, bin_img


def adaptiveMeanThresholding(img_obj, **kwargs):
    max_val = kwargs.get("max_val", 255)
    inv = cv2.THRESH_BINARY_INV if kwargs.get("invert") else cv2.THRESH_BINARY
    block_size = kwargs.get("block_size", 3)
    c_const = kwargs.get("c_const", 2)

    return cv2.adaptiveThreshold(getGrayImg(img_obj), max_val, cv2.ADAPTIVE_THRESH_MEAN_C, inv, block_size, c_const)


def adaptiveGaussianThresholding(img_obj, **kwargs):
    max_val = kwargs.get("max_val", 255)
    inv = cv2.THRESH_BINARY_INV if kwargs.get("invert") else cv2.THRESH_BINARY
    block_size = kwargs.get("block_size", 3)
    c_const = kwargs.get("c_const", 2)

    return cv2.adaptiveThreshold(getGrayImg(img_obj), max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, inv, block_size, c_const)


def otsuThresholding(img_obj, **kwargs):
    thr_val = kwargs.get("thr_val", 120)
    max_val = kwargs.get("max_val", 255)
    inv = cv2.THRESH_BINARY_INV if kwargs.get("invert") else cv2.THRESH_BINARY

    ret, bin_img = cv2.threshold(getGrayImg(img_obj), thr_val, max_val, inv + cv2.THRESH_OTSU)

    return ret, bin_img


def gaussianBlurAndOtsuThresholding(img_obj, **kwargs):
    kernel_size = kwargs.get("kernel_size", (5,5))
    thr_val = kwargs.get("thr_val", 120)
    max_val = kwargs.get("max_val", 255)
    inv = cv2.THRESH_BINARY_INV if kwargs.get("invert") else cv2.THRESH_BINARY

    blur_img = cv2.GaussianBlur(getGrayImg(img_obj), kernel_size, 0)
    ret, bin_img = cv2.threshold(blur_img, thr_val, max_val, inv + cv2.THRESH_OTSU)

    return ret, bin_img


def changeAlphaBeta(img_obj, alpha, beta):
    if alpha < 1. or alpha > 3.: return False
    if beta < 0 or alpha > 100: return False
    return cv2.convertScaleAbs(img_obj, alpha=alpha, beta=beta)


def linearContrast(img_obj, **kwargs):
    percentage = kwargs.get('percentage', 0)
    angle = mapValue(var=percentage, var_min=0, var_max=100, ret_min=0, ret_max= 90)
    altered_img = copy.copy(img_obj)

    table_pixels = {ind: np.uint8(np.clip(ind*math.tan(angle * (math.pi / 180)), 0, 255))
                    for ind in range(256)}

    rows, cols, layers = altered_img.shape
    for row_ind in range(rows):
        for col_ind in range(cols):
            for layer_ind in range(layers):
                altered_img[row_ind, col_ind, layer_ind] = table_pixels[altered_img[row_ind][col_ind][layer_ind]]

    return altered_img


def logaritmicContrast(img_obj):
    altered_img = copy.copy(img_obj)

    max_val = np.max(altered_img)
    table_pixels = {ind: np.uint8(np.clip((255/np.log10(max_val))*np.log10(ind+1), 0, 255))
                    for ind in range(256)}

    rows, cols, layers = altered_img.shape
    for row_ind in range(rows):
        for col_ind in range(cols):
            for layer_ind in range(layers):
                altered_img[row_ind, col_ind, layer_ind] = table_pixels[altered_img[row_ind][col_ind][layer_ind]]

    return altered_img


def variableLinearContrast(img_obj, angle):
    altered_img = copy.copy(img_obj)

    table_pixels = {}
    for ind in range(256):
        if ind <= 78:
            angle, b = 25, 0 #makes the dark pixels darker
        elif ind > 78 and ind < 136:
            angle, b = 65, -120 #makes the pixels in the middle brighter
        elif ind >= 136:
            angle, b = 20, 162 #makes the brighter pixels darker

        table_pixels[ind] = np.uint8(np.clip(ind*math.tan(angle * (math.pi / 180)) + b, 0, 255))

    rows, cols, layers = altered_img.shape
    for row_ind in range(rows):
        for col_ind in range(cols):
            for layer_ind in range(layers):
                altered_img[row_ind, col_ind, layer_ind] = table_pixels[altered_img[row_ind][col_ind][layer_ind]]

    return altered_img


def quadraticContrast(img_obj, **kwargs):
    altered_img = copy.copy(img_obj)
    a_const = 10**-2.4066 if (percentage := kwargs.get('percentage')) == None else 10**(mapValue(percentage, 0, 100, -5, -1))

    table_pixels = {ind: np.uint8(np.clip(a_const*(ind**2), 0, 255))
                    for ind in range(256)}

    rows, cols, layers = altered_img.shape
    for row_ind in range(rows):
        for col_ind in range(cols):
            for layer_ind in range(layers):
                altered_img[row_ind, col_ind, layer_ind] = table_pixels[altered_img[row_ind][col_ind][layer_ind]]

    return altered_img


def exponencialContrast(img_obj, **kwargs):
    altered_img = copy.copy(img_obj)
    a_const = 46 if (percentage := kwargs.get('percentage')) == None else mapValue((100-percentage), 0, 100, 10, 65)

    table_pixels = {ind: np.uint8(np.clip(np.exp(ind/a_const)-1, 0, 255))
                    for ind in range(256)}

    rows, cols, layers = altered_img.shape
    for row_ind in range(rows):
        for col_ind in range(cols):
            for layer_ind in range(layers):
                altered_img[row_ind, col_ind, layer_ind] = table_pixels[altered_img[row_ind][col_ind][layer_ind]]

    return altered_img


def sqrtContrast(img_obj, **kwargs):
    altered_img = copy.copy(img_obj)
    a_const = 16 if (percentage := kwargs.get('percentage')) == None else mapValue(percentage, 0, 100, 1, 30)

    table_pixels = {ind: np.uint8(np.clip(a_const*np.sqrt(ind), 0, 255))
                    for ind in range(256)}

    rows, cols, layers = altered_img.shape
    for row_ind in range(rows):
        for col_ind in range(cols):
            for layer_ind in range(layers):
                altered_img[row_ind, col_ind, layer_ind] = table_pixels[altered_img[row_ind][col_ind][layer_ind]]

    return altered_img


def bezierContrast(img_obj, **kwargs):
    altered_img = copy.copy(img_obj)

    list_points = kwargs.get('list_points', [[0.72, 0], [0.26, 1]])
    points = [[0,0], list_points[0], list_points[1], [1,1]]

    _, yvals = bezier.bezier_curve(points, nTimes=256)
    table_pixels = {ind: np.uint8(ind * yvals[ind]) for ind in range(256)}

    rows, cols, layers = altered_img.shape
    for row_ind in range(rows):
        for col_ind in range(cols):
            for layer_ind in range(layers):
                altered_img[row_ind, col_ind, layer_ind] = table_pixels[altered_img[row_ind][col_ind][layer_ind]]

    return altered_img


def showGrayImgsHistogram(img_obj_list, **kwargs):
    num_bins = kwargs.get("num_bins", 256)
    hist_range = kwargs.get("hist_range", (0, 256))
    mask = kwargs.get("mask", None)

    plt.figure(figsize=(10, 5))
    plt.suptitle('Gray-scale image histogram')
    num_lines = len(img_obj_list)
    for ind, img in enumerate(img_obj_list, start=1):
        gray_img = getGrayImg(img)
        gray_hist = cv2.calcHist([gray_img], [0], mask, [num_bins], hist_range)

        plt.subplot(num_lines, 2, (2*ind)-1)
        plt.imshow(gray_img, cmap='gray')
        plt.title('Image')
        plt.axis('off')

        plt.subplot(num_lines, 2, (2*ind))
        plt.title('Gray Image Histogram')
        plt.xlabel('Bins'), plt.ylabel('# of pixels')
        plt.plot(gray_hist, color='gray')
        plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()


def showColoredImgshistogram(img_obj_list, **kwargs):
    layers2show = kwargs.get("layers", [0, 1, 2])
    num_bins = kwargs.get("num_bins", 256)
    hist_range = kwargs.get("hist_range", (0, 256))
    mask = kwargs.get("mask", None)

    plt.figure(figsize=(10, 5))
    plt.suptitle('Colored Images Histogram')
    num_lines = len(img_obj_list)
    colors = ['b', 'g', 'r']
    for ind, img in enumerate(img_obj_list, start=1):
        bgr_layers = cv2.split(img)

        plt.subplot(num_lines, 2, (ind*2)-1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_img)
        plt.title('Colored Image')
        plt.axis('off')

        plt.subplot(num_lines, 2, ind*2)
        for ind in range(len(layers2show)):
            hist = cv2.calcHist(bgr_layers, [ind], mask, [num_bins], hist_range)
            plt.plot(hist, color=colors[ind])
        plt.title('Colors Histogram')
        plt.xlabel('Pixel Intensity'), plt.ylabel('# of pixels')
        plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()


def equalizeHist(img_obj, **kwargs):
    lab_img = cv2.cvtColor(img_obj, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    equa = cv2.equalizeHist(l)

    equalized_lab_img = cv2.merge((equa, a, b))
    equalized_img = cv2.cvtColor(equalized_lab_img, cv2.COLOR_LAB2BGR)

    if kwargs.get('show_imgs'):
        showImages([img_obj, equalized_img], ['Original Image', 'Equalized Image'])
    elif kwargs.get('show_imgs_hist'):
        showColoredImgshistogram([img_obj, equalized_img])

    return equalized_img


def CLAHE(img_obj):
    lab_img = cv2.cvtColor(img_obj, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab_img)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(l)

    clahe_lab_img = cv2.merge((clahe_img, a, b))
    clahe_img = cv2.cvtColor(clahe_lab_img, cv2.COLOR_LAB2BGR)

    return clahe_img


def getContours(th):
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(th, rect_kern, iterations=1)

    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])


if __name__ == '__main__':
    lenna = cv2.imread('../test_images/lenna.png', cv2.IMREAD_COLOR)
    fro = cv2.imread('../test_images/sit_frog.jpg', cv2.IMREAD_COLOR)
    page = cv2.imread('../test_images/page.jpg', cv2.IMREAD_COLOR)
    gray_fro = cv2.imread('../test_images/sit_frog.jpg', cv2.IMREAD_GRAYSCALE)

    '''_, absolute = plainAbsoluteThresholding(page)
    mean = adaptiveMeanThresholding(page)
    gaussian = adaptiveGaussianThresholding(page)
    _, otsu = otsuThresholding(page)
    _, otsuGaussian = gaussianBlurAndOtsuThresholding(page)
    showImages([absolute, mean, gaussian, otsu, otsuGaussian],
               ['absolute', 'mean', 'gaussian', 'otsu', 'otsuGaussian'], num_rows=2)'''

    showImages([fro, sqrtContrast(fro), bezierContrast(fro)], ['orig', 'squaere', 'bezier'])
