import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from mpl_toolkits.mplot3d import Axes3D

def FreqFilter(img, isGray):
    if ~isGray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    F2 = np.fft.fft2(img.astype(float))
    Y = (np.linspace(-int(F2.shape[0]/2), int(F2.shape[0]/2)-1, F2.shape[0]))
    X = (np.linspace(-int(F2.shape[1]/2), int(F2.shape[1]/2)-1, F2.shape[1]))
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, np.fft.fftshift(np.abs(F2)), cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Z = np.fft.fftshift(np.log(np.abs(F2)+1))    
    ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False) 
    plt.show()



    magnitudeImage = np.fft.fftshift(np.abs(F2))
    magnitudeImage = magnitudeImage / magnitudeImage.max()   # scale to [0, 1]
    magnitudeImage = ski.img_as_ubyte(magnitudeImage)
    cv2.imwrite('ManitudeImg.JPG', magnitudeImage)

    logMagnitudeImage = np.fft.fftshift(np.log(np.abs(F2)+1))
    logMagnitudeImage = logMagnitudeImage / logMagnitudeImage.max()   # scale to [0, 1]
    logMagnitudeImage = ski.img_as_ubyte(logMagnitudeImage)
    cv2.imwrite('LogManitudeImg.JPG', logMagnitudeImage)


def FreqAnalysis(img,isGray):
    if ~isGray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    colData = img[:, 0]
    xvalues = np.linspace(0, len(colData) - 1, len(colData))
    plt.plot(xvalues, colData, 'b')
    F_colData = np.fft.fft(colData.astype(float))
    plt.figure()
    xvalues = np.linspace(-len(colData), len(colData), len(colData))
    markerline, stemlines, baseline = plt.stem(xvalues, np.fft.fftshift(np.log(np.abs(F_colData)+1)), 'g')
    plt.setp(markerline, 'markerfacecolor', 'g')
    plt.setp(baseline, 'color','r', 'linewidth', 0.5)
    plt.show()
    
def reverseFourier(img,isGray):
    if ~isGray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    F=np.fft.fft2(img.astype(float))
    #I = np.fft.ifft(F)
    #cv2.imwrite('fuuuuuu.JPG',I.real)   
    k = 20
    # Truncate frequencies and then plot the resulting function in real space
    Trun_F_Data = F.copy()
    Trun_F_Data[int(img.shape[0]/2 -k):int( img.shape[0]/2 +k),int(img.shape[1]/2 -k):int(img.shape[0]/2 +k)] = 0
    trun_Data = np.fft.ifft(Trun_F_Data)
    xvalues = np.linspace(0, trun_Data.shape[0] - 1, trun_Data.shape[1])
    yvalues = np.linspace(0, trun_Data.shape[1] - 1, trun_Data.shape[0]) 
    X, Y = np.meshgrid(xvalues, yvalues)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.plot_surface(X, Y, img)
    ax.plot_surface(X,Y, img)
    ax.plot_surface(X, Y, trun_Data.real,cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)

    plt.title('k = 0 : ' + str(k))
    plt.show()
    plt.clf()
    
def reverseFourier1D(img,isGray):
    if ~isGray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    colData = img[:,int(img.shape[1]/2)]
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    # Compute the 1-D Fourier transform of colData
    F_colData = np.fft.fft(colData.astype(float))
    for k in range(200, 205):
        # Truncate frequencies and then plot the resulting function in real space
        Trun_F_colData = F_colData.copy()
        Trun_F_colData[k+1:len(Trun_F_colData)-k] = 0
        trun_colData = np.fft.ifft(Trun_F_colData)
        # Plot
        xvalues = np.linspace(0, len(trun_colData) - 1, len(trun_colData))
        plt.plot(xvalues, colData, 'b')
        plt.plot(xvalues, trun_colData, 'r')
        plt.title('k = 0 : ' + str(k))
        plt.show()
        plt.clf()
    for k in range(200, 205):
        # Truncate frequencies and then plot the resulting function in real space
        Trun_F_colData = F_colData.copy()
        Trun_F_colData[:k+1] = 0
        Trun_F_colData[len(Trun_F_colData)-k:] = 0
        trun_colData = np.fft.ifft(Trun_F_colData)
        # Plot
        xvalues = np.linspace(0, len(trun_colData) - 1, len(trun_colData))
        plt.plot(xvalues, colData, 'b')
        plt.plot(xvalues, trun_colData, 'r')
        plt.title('k = 0 : ' + str(k))
        plt.show()
        plt.clf()

def HighLowPass(img,isGray):
    if ~isGray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    U = (np.linspace(-int(img.shape[0]/2), int(img.shape[0]/2)-1, img.shape[0]))

    V = (np.linspace(-int(img.shape[1]/2), int(img.shape[1]/2)-1, img.shape[1]))
    U, V = np.meshgrid(V, U)
    D = np.sqrt(V*V + U*U)
    xval = np.linspace(-int(img.shape[1]/2), int(img.shape[1]/2)-1, img.shape[1])
    D0 = 0.25 * D.max()
    D1 = 0.75 * D.max()
    idealLowPass = D <= D0
    idealHighPass = D >= D1
    # Filter our small grayscale image with the ideal lowpass filter
    # 1. DFT of image
    FTgraySmall = np.fft.fft2(img.astype(float))
    FTgraySmallFiltered = FTgraySmall * np.fft.fftshift(idealLowPass)
    # 4. Inverse DFT to take filtered image back to the spatial domain
    graySmallFiltered = np.abs(np.fft.ifft2(FTgraySmallFiltered))
    
    # Save the filter and the filtered image (after scaling)
    idealLowPass = ski.img_as_ubyte(idealLowPass / idealLowPass.max())
    graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())
    cv2.imwrite("idealLowPass.jpg", idealLowPass)
    cv2.imwrite("Low_grayImageIdealLowpassFiltered.jpg", graySmallFiltered)
    
    # Plot the ideal filter and then create and plot Butterworth filters of order
    # n = 1, 2, 3, 4
    plt.plot(xval, idealLowPass[int(idealLowPass.shape[0]/2), :], 'c--', label='ideal')
    colors='brgkmc'
    for n in range(1, 5):
        # Create Butterworth filter of order n
        H = 1.0 / (1 + (np.sqrt(2) - 1)*np.power(D/D0, 2*n))
        # Apply the filter to the grayscaled image
        FTgraySmallFiltered = FTgraySmall * np.fft.fftshift(H)
        graySmallFiltered = np.abs(np.fft.ifft2(FTgraySmallFiltered))
        graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())
        cv2.imwrite("Low_grayImageButterworth-n" + str(n) + ".jpg", graySmallFiltered)
        # cv2.imshow('H', H)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        H = ski.img_as_ubyte(H / H.max())
        cv2.imwrite("butter-n" + str(n) + ".jpg", H)
        # Get a slice through the center of the filter to plot in 2-D
        slice = H[int(H.shape[0]/2), :]
        plt.plot(xval, slice, colors[n-1], label='n='+str(n))
        plt.legend(loc='upper left')
    
    # plt.show()
    plt.savefig('Low_butterworthFilters.jpg', bbox_inches='tight')
    
    FTgraySmall = np.fft.fft2(img.astype(float))

    FTgraySmallFiltered = FTgraySmall * np.fft.fftshift(idealHighPass)
    # 4. Inverse DFT to take filtered image back to the spatial domain
    graySmallFiltered = np.abs(np.fft.ifft2(FTgraySmallFiltered))
    
    # Save the filter and the filtered image (after scaling)
    idealHighPass = ski.img_as_ubyte(idealHighPass / idealHighPass.max())
    graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())
    cv2.imwrite("idealHighass.jpg", idealHighPass)
    cv2.imwrite("High_grayImageIdealHighpassFiltered.jpg", graySmallFiltered)
    
    # Plot the ideal filter and then create and plot Butterworth filters of order
    # n = 1, 2, 3, 4
    plt.plot(xval, idealHighPass[int(idealHighPass.shape[0]/2), :], 'c--', label='ideal')
    colors='brgkmc'
    for n in range(1, 5):
        # Create Butterworth filter of order n
        H = 1.0 / (1 + (np.sqrt(2) - 1)*np.power(D/D0, 2*n))
        # Apply the filter to the grayscaled image
        FTgraySmallFiltered = FTgraySmall * np.fft.fftshift(H)
        graySmallFiltered = np.abs(np.fft.ifft2(FTgraySmallFiltered))
        graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())
        cv2.imwrite("High_grayImageButterworth-n" + str(n) + ".jpg", graySmallFiltered)
        # cv2.imshow('H', H)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        H = ski.img_as_ubyte(H / H.max())
        cv2.imwrite("butter-n" + str(n) + ".jpg", H)
        # Get a slice through the center of the filter to plot in 2-D
        slice = H[int(H.shape[0]/2), :]
        plt.plot(xval, slice, colors[n-1], label='n='+str(n))
        plt.legend(loc='upper left')
    
    # plt.show()
    plt.savefig('High_butterworthFilters.jpg', bbox_inches='tight')
    
    
def main():
    img = cv2.imread('OrigPuppy.JPG')
    img2 = cv2.imread('LogManitudeImg.JPG')
    
    FreqFilter(img,False)
    FreqAnalysis(img,False)
    reverseFourier(img,False)
    reverseFourier1D(img,False)
    
    HighLowPass(img,False)
    

if __name__ == '__main__':
    main()