from PIL import Image
import matplotlib.pyplot  as plt
import numpy as np
import cv2;
import scipy as misc
import operator;
import glob
import os;

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 32
RESIZED_IMAGE_HEIGHT = 32

strFinalString = ""
npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
validContoursData = []
intClassifications = []
npaClassifications = None
arabChars = []
kNearest = None

for x in range(0, 36):
    arabChars.append(1575 + x)
    d = arabChars[x]
    print(chr(d), " unicode:", arabChars[x])

def TrainFlatCharData(img):
    charImg = img.reshape(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT)

    imgThresh = PreProcessChar(charImg, True)
    TrainCharData(imgThresh)

def TestFlatCharData(img):
    charImg = img.reshape(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT)

    imgThresh = PreProcessChar(charImg, True)
    TestCharData(imgThresh)

def PreProcessChar(img, isGray = False):    
    if(isGray is False):
        img = cv2.CvtColor(img, cv2.CV_GRAY2BGR)

    imgBlurred = cv2.GaussianBlur(img, (1,1), 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    return imgThresh

def TrainCharData(imgThresh):

    global intClassifications, npaFlattenedImages

    imgThreshCopy = imgThresh.copy()
    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)          

    for npaContour in npaContours:                          
        if (cv2.contourArea(npaContour) >= MIN_CONTOUR_AREA):          
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)        
                                            
            cv2.rectangle(imgThresh, (intX, intY), (intX+intW,intY+intH), (0, 0, 255), 2)        
            
            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]                                 
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     

            cv2.imshow("Train Image", imgThresh)  
            cv2.waitKey(500)
            cv2.destroyAllWindows()

            intChar = ord(input("What's this ?"))

            print("Pressed: ", chr(intChar), " Unicode: ", intChar)
            
            if intChar == 27: 
                return                      
            elif intChar in arabChars:      
                intClassifications.append(intChar)                                               

                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)                    

def TestCharData(imgThresh):
    global strFinalString

    imgThreshCopy = imgThresh.copy()        

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   

    for npaContour in npaContours:                            
        contourData = ContourData(npaContour)                                             
        if contourData.IsValidContour:          
            validContoursData.append(contourData)                                     

    validContoursData.sort(key = operator.attrgetter("intRectX"))
    cv2.rectangle(imgThresh, (contourData.intRectX, contourData.intRectY), (contourData.intRectX + contourData.intRectWidth, contourData.intRectY + contourData.intRectHeight), (0, 255, 0), 2)                        

    imgROI = imgThresh[contourData.intRectY : contourData.intRectY + contourData.intRectHeight,     
                        contourData.intRectX : contourData.intRectX + contourData.intRectWidth]

    imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             

    npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))     

    npaROIResized = np.float32(npaROIResized)   
    
    retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)     
     
    strCurrentChar = chr(npaResults[0][0])                                            

    print("Guess: ", strCurrentChar, " Unicode : ", npaResults[0][0])

    cv2.imshow("RES", imgROI)

    cv2.waitKey(0)
    
    strFinalString = strFinalString + strCurrentChar  

def GeneratekNNFiles():
    global intClassifications, npaClassifications, fltClassifications, npaFlattenedImages

    fltClassifications = np.array(intClassifications, np.float32)    
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))   

    np.savetxt("classifications.txt", npaClassifications)         
    np.savetxt("flattened_images.txt", npaFlattenedImages) 

class ContourData():
    npaContour = None
    boundingRect = None
    intRectX = 0
    intRectY = 0
    intRectWidth = 0
    intRectHeight = 0
    fltArea = 0.0

    def __init__(self, npaC = None):
        if(npaC is None):
            print("No contour data were to set to this ContourData")
            return

        #print("ContourData Object Created")

        self.npaContour = npaC
        self.boundingRect = cv2.boundingRect(self.npaContour)
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight
        self.fltArea = cv2.contourArea(self.npaContour)

    def IsValidContour():
        if (self.fltArea >= MIN_CONTOUR_AREA):
            return True
        return False

def main():
    global npaClassifications, npaFlattenedImages, kNearest
    
    print("\nWish to train your data? \nPress Y to confirm.")
    cv2.imshow("",np.arange(0,1))
    confKey = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if confKey ==  89:
        print("\nTraining Data...")

        trainfilelist = glob.glob("realFewTrain/*.png")            #Get all train file paths in a list
        npaTrainImages = np.array([np.array(Image.open(fname)) for fname in trainfilelist])         #Get all the images in a Numpy array
        npaTrainImages = npaTrainImages.reshape(-1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)            #Make each row a flattened image

        np.apply_along_axis(TrainFlatCharData, 1, npaTrainImages)

        GeneratekNNFiles()
        cv2.destroyAllWindows()     

    print("\nTraining flattened images with classification values...")

    npaClassifications = np.loadtxt("classifications.txt", np.float32)
    npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)

    print("Flattened Images: \n", npaFlattenedImages.shape)
    print("Classifier: \n", npaClassifications.shape)
        
    print("\nCreating kNN Object...")

    kNearest = cv2.ml.KNearest_create()          

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    print("\nTesting Data...")

    testfilelist = glob.glob("realFewTest/*.png")           
    npaTestImages = np.array([np.array(Image.open(fname)) for fname in testfilelist])        
    npaTestImages = npaTestImages.reshape(-1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)            

    np.apply_along_axis(TestFlatCharData, 1, npaTestImages)

    print ("\n" + strFinalString + "\n")            

    cv2.waitKey(0)

if __name__ == "__main__":
    main()

  