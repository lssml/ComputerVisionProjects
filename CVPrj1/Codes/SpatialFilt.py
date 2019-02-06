import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

def SpatialFilter (img, filt,filtname, outputname):
    convolved = np.zeros(img.shape, np.float64)  # array for denoised image
    # Apply the filter to each channel
    convolved[:, :, 0] = cv2.filter2D(img[:, :, 0], -1, filt)
    convolved[:, :, 1] = cv2.filter2D(img[:, :, 1], -1, filt)
    convolved[:, :, 2] = cv2.filter2D(img[:, :, 2], -1, filt)
    #convolved = convolved / img.max() *255  
    convolved = convolved.astype(np.uint8)

    cv2.imwrite(filtname, filt*255)
    cv2.imwrite(outputname , convolved) 
    
    cv2.imshow('original', img)
    cv2.imshow('filter', filt)
    cv2.imshow('convolved', convolved)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    



def getGaussFilt(shape,sigma):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def MedianFilter(img, outputname, k):
    median = cv2.medianBlur(img,k)
    cv2.imwrite(outputname , median)

    
def CannyFilter(img, outputname):
    edge = cv2.Canny(img,100,200)
    cv2.imwrite(outputname, edge)

def main():
    img = cv2.imread('Puppy.JPG')
    img2 = cv2.imread('OrigPuppy.JPG')
    img3 = cv2.imread('Forest.JPG')
    img4 = cv2.imread('123.JPG')
    
    for k in [3,9,27]:
        GaussFilt = getGaussFilt((k,k),5)
        #GaussFilt = np.zeros((10,10))
        SpatialFilter(img, GaussFilt,str(k)+'GaussFilter.JPG',str(k)+'Gauss_Puppy.JPG')
        MedianFilter(img, str(k)+'Median_Puppy.JPG', k)
    
    CannyFilter(img, 'NoisyPuppyCanny.JPG')
    CannyFilter(img2, 'OrigPuppyCanny.JPG')
    CannyFilter(img3, 'ForestCanny.JPG')
    

if __name__ == '__main__':
    main()
    