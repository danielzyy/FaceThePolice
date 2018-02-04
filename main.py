import pygame
import pygame.camera
import face_recognition
import cv2
pygame.init()
pygame.camera.init()
bigCalibri=pygame.font.SysFont ("Calibri",30)
smallCalibri=pygame.font.SysFont("Calibri",20)
gui=pygame.image.load('gui.png')
subjectHere=pygame.image.load('subjectHere.png')

RED= (255,0,0)
BLACK=(0,0,0)
WHITE= (255,255,255)
GREEN= (0,255,0)
BLUE= (0,0,255)
SKYBLUE=(122,214,245)
BEIGE= (186,111,26)
PINK=(255,0,238)
ORANGE=(255,165,0)
PEACH= (232, 135, 78)
LIGHTGREEN= (177,233,111)
GREY= (52, 54, 58)
BROWN=(86,62,38)  #Color Constants

#database
def readDatabase():
    pathList = []
    names = []
    crimes = []
    database = open("existingdata.dat",'r')
    while True:
        dataline = database.readline()
        if dataline == '':
            break
        dataline = dataline.replace('\n','')
        record = dataline.split(';')
        pathList.append("/home/deeplearning/Desktop/HACKATHON/existingfaces/" + record[0])
        names.append(record[1])
        crimes.append(record[2])
    database.close()
    return pathList,names,crimes

def compileImages(pathList):
    imageList = []
    for path in pathList:
        print (path)
        imageList.append(face_recognition.load_image_file(path))
    return imageList

def simpleCompare(camEncoding,refEncodings):
    compareArrays = face_recognition.compare_faces(refEncodings,camEncoding,tolerance=0.105)
    matchBoolList = []
    for Array in compareArrays:
        count = 0
        for element in Array:
            if element == False:
                count += 1
        if count < 2:
            matchBoolList.append(True)
        else:
            matchBoolList.append(False)
    return matchBoolList

def targetDataBlit(nonVolitileListPosition, nonVolitileListLocal,imageListLocal,nameListLocal,crimesListLocal):
    localIndex = nonVolitileListLocal[nonVolitileListPosition]
    screen.blit(subjectHere,(640,0,200,400))
    screen.blit(imageListLocal[localIndex],(640,0))
    localName=bigCalibri.render(nameListLocal[localIndex],False,(255,0,0))
    screen.blit(localName,(640,155))
    localCrimes = crimesListLocal[localIndex]
    crimesLines = []
    crimesLines = localCrimes.split("*")
    y = 190
    for line in crimesLines:
        crimeBuffer = smallCalibri.render(line,False,(255,0,0))
        screen.blit(crimeBuffer,(640,y))
        y += 15

def infoCollisions(x, y, mouseButton,nonVolitileListLocal, indexForNonVolitileList):
    mouseHitRect=pygame.Rect(x,y,1,1)
    actionList=[(640,400,100,40),(740,400,100,40),(640,440,100,40),(740,440,100,40)]
    collide=mouseHitRect.collidelist(actionList)
    if collide != -1 and mouseButton==1:
        if collide==0:
            nonVolitileListLocal=[]
        elif collide==1:
            addData(getLastKey())
        elif collide==3 and indexForNonVolitileList < len(nonVolitileListLocal)-1:
            indexForNonVolitileList+=1
        elif collide==2 and indexForNonVolitileList > 0:
            indexForNonVolitileList-=1
    return nonVolitileListLocal, indexForNonVolitileList

def loadAllImagesIntoSurfaces(pathList):
    localImageList = []
    for path in pathList:
        img = pygame.image.load(path)
        img = pygame.transform.scale(img,(200,150))
        localImageList.append(img)
    return localImageList
def getVal(tup):
    """ getVal returns the (position+1) of the first 1 within a tuple.
        This is used because MOUSEBUTTONDOWN and MOUSEMOTION deal with
        mouse events differently
    """
    for i in range(3):
        if tup[i]==1:
            return i+1
    return 0

def getLastKey():
    pathList = []
    database = open("existingdata.dat",'r')
    while True:
        dataline = database.readline()
        if dataline == '':
            break
        dataline = dataline.replace('\n','')
        record = dataline.split(';')[0]
        pathList.append(record)
    return pathList[-1]

def addData(lastKey):
    lastKey = lastKey.replace('.jpg','')
    record = str(int(lastKey) + 1) + '.jpg;'
    camFrame = pygame.image.load("/home/deeplearning/videoFrame.jpg")
    fileLocation = '/home/deeplearning/Desktop/HACKATHON/existingfaces/'+record.replace(';','')
    pygame.image.save(camFrame,fileLocation)
    name = input("Please enter the person's name: ")
    if name == '':
        return ''
    record += name + ';'
    while True:
        crime = input("Please enter brief descriptions of the person's offense (END to stop): ")
        if crime == "END":
            break
        record += crime + '*'
    record = record[0:len(record)-1]
    finalReferenceEncodings.append(face_recognition.face_encodings(face_recognition.load_image_file(fileLocation)))
    nameList.append(name)
    crimessList.append(record.split(';')[2])
    database = open("existingdata.dat",'a')
    database.write(record)
    database.close()

targetList,nameList,crimesList=readDatabase()
targetFaceList=compileImages(targetList)
finalReferenceEncodings=[]
for image in targetFaceList:
    finalReferenceEncodings.append(face_recognition.face_encodings(image))
pygameImages = loadAllImagesIntoSurfaces(targetList)
thresholdCounter = 0

screen = pygame.display.set_mode((840,480))
clock = pygame.time.Clock()
wifiCam = pygame.camera.Camera('/dev/video0',(640,480))
wifiCam.start()
majorityLists=[[]]
nonVolitileList=[]
blitCounter = 0
running=True

while running:
    button=0
    mx=-1
    my=-1
    for evnt in pygame.event.get():
        if evnt.type==pygame.QUIT:
            running=False
        elif evnt.type==pygame.MOUSEBUTTONDOWN:
            mx,my=evnt.pos
            button=evnt.button
        elif evnt.type==pygame.MOUSEMOTION:
            button = getVal(evnt.buttons)
            mx,my=evnt.pos

    imageSurface = wifiCam.get_image()
    pygame.image.save(imageSurface,"/home/deeplearning/videoFrame.jpg")
    frame = face_recognition.load_image_file("/home/deeplearning/videoFrame.jpg",mode='RGB')
    screen.blit(imageSurface,(0,0))
    locationBoxes = face_recognition.face_locations(frame)
    #print (locationBoxes)#Face locations are rectangles (top,right,bottom,left)
    encodings = face_recognition.face_encodings(frame,locationBoxes)
    for i in range(len(encodings)):
        #print (face_recognition.compare_faces(finalReferenceEncodings,encodings[i],tolerance=0.105))
#ACTUAL BREAK
        if len(majorityLists) < i + 1:
            majorityLists.append([])
        matchBools = simpleCompare(encodings[i],finalReferenceEncodings)
        if True in matchBools:
            color = (255,0,0)
            majorityLists[i].append(nameList[matchBools.index(True)])
        else:
            color = (255,255,255)
            majorityLists[i]=[]
        if len(majorityLists[i]) > 5:
            dataIndex = matchBools.index(True)
            "[font render] n40ameList[dataIndex]"
            "[font render] crimesList[dataIndex]"
            majorityCounter=0
            for name in nameList:
                temp=majorityLists[i].count(name)
                if temp > majorityCounter:
                    majorityCounter=temp
                    dataIndex=nameList.index(name)
            if dataIndex not in nonVolitileList:
                nonVolitileList.append(dataIndex)
            majorityLists[i] = majorityLists[i][1:]
            print (majorityLists[i])
            print (nonVolitileList)

        pygame.draw.rect(screen,color,(locationBoxes[i][3],locationBoxes[i][0],abs(locationBoxes[i][2]-locationBoxes[i][0]),abs(locationBoxes[i][1]-locationBoxes[i][3])),5)
    nonVolitileList, blitCounter = infoCollisions(mx, my, button,nonVolitileList, blitCounter)
    print(blitCounter)
    if nonVolitileList != []:
        targetDataBlit(blitCounter, nonVolitileList,pygameImages,nameList,crimesList)
    else:
        screen.blit(subjectHere,(640,0,200,400))
        blitCounter = 0
         #placeholder rectangle
    screen.blit(gui,(640,400,100,100))
    pygame.display.flip()
    clock.tick(20)
wifiCam.stop()
# pygame.quit()
