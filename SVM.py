"""
Created on Sun Jun 27 23:36:29 2021

@author: muhammaduzair
"""

#Libraries used 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split


#Load the dataset
data=np.load("data/Training_faces.npy")
target=np.load("data/Testing_faces.npy")

#Split the dataset into training and testing data.
#test_size=0.20 means testing data is 20% and training data is 80%
trainX, testX, trainY, testY = train_test_split(data, target, test_size=0.20, random_state=42)


#Find the size of the training dataset
Size=trainX.shape
#z means total number of images present in training data
z=Size[0]
row=Size[1]
col=Size[2]

Train=np.array(trainX)
#Reshape the data to 2d array
TrainX=Train.reshape(-1,col*row)
#convert data to double
TrainX = TrainX/255


while 1:
    View_Image = int(input("Enter Image Number to view:"))
    if View_Image>z:
        print("Image Number not found")
        print("The Image Number must be between 0 to ",z-1)
    else:
        break

#See any image present in the data
plt.figure(1)
plt.imshow(trainX[View_Image])
plt.title('Example image from training set');
plt.show()


#Procedure for image reconstruction after applying PCA
trainMean = TrainX.mean(axis=0)
trainMean=trainMean.transpose()
train_tilde = TrainX-trainMean;

#PCA variance selection
pca = PCA(.95)
comp=pca.fit(train_tilde)
pca.fit(TrainX)
Trained=pca.transform(TrainX)
Comp=comp.components_.transpose()
F=train_tilde.dot(Comp)

#Image reconstructed after PCA
trainX_Reconstructed=F.dot(Comp.transpose())

#Reshape image to view
Trained_X=trainX_Reconstructed.reshape((z,col,row))

#View the reconstructed Image
plt.figure(2)
plt.imshow(Trained_X[View_Image])
plt.title("Selected Image after reconstruction")
plt.show()


#Get the size of testing dataset
Size1=testX.shape

#z1 is the total number of images present in testing dataset
z1=Size1[0]
row1=Size1[1]
col1=Size1[2]


Test=np.array(testX)
#Reshape the testing image to 2d array
TestX=Test.reshape(-1,col*row)
#Convert image to double
TestX = TestX/255

#Procedure for reconstruction of image after applying PCA 
testMean = TestX.mean(axis=0)
testMean=testMean.transpose()
test_tilde = TestX-testMean;
test_k = test_tilde.dot(Comp);
New_featured_data_testing = test_k.dot(Comp.transpose())
Testing=pca.transform(TestX)

#Train the classifier using SVM classifier on Training data
print("Training the SVM Model.")
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(Trained, trainY)


#Estimate the accuracy of the classifier on future data, using the test data
print("Training Done.")
print("Accuracy of the Model=",clf.score(Testing , testY)*100,"%")

#Check the model result by applying any image

while 1:
    Test_Image=int(input("Enter Image Number to Test:"))

    if Test_Image>=z1:
        print("Image Number not found")
        print("Image Number must be between 0 to ",z1-1)
    else:
        break
    
#Uncomment bewlo line if you want to see reconstructed testing image
#Test=New_featured_data_testing[Test_Image,:] 
#Comment below line if you want to see reconstructed testing image
Test=testX[Test_Image,:]
Testing_Image=Test.reshape(row,col)
print("Testing Image Separately")
Person="Selected Person Number = "+(str(testY[Test_Image]))

plt.figure()
plt.imshow(Testing_Image)
plt.title("Image Selected For Testing")

plt.xlabel(Person)
plt.show()

#Predicting the testing person
Test=Testing[Test_Image,:]
Test=Test.reshape((-1, 1))
Test=Test.transpose()
Pred=clf.predict(Test)
print("Prediction:")
print("The Testing Person Number Is =",Pred)
