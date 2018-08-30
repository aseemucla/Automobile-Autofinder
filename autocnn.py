import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import imagetoarr

#For Google Colab

# from google.colab import auth
# auth.authenticate_user()
# from googleapiclient.discovery import build
# drive_service = build('drive', 'v3')

# file_id = 'target_file_id'

# import io
# from googleapiclient.http import MediaIoBaseDownload

# request = drive_service.files().get_media(fileId=file_id)
# downloaded = io.BytesIO()
# downloader = MediaIoBaseDownload(downloaded, request)
# done = False
# while done is False:
#   # _ is a placeholder for a progress object that we ignore.
#   # (Our file is small, so we skip reporting progress.)
#   _, done = downloader.next_chunk()

# downloaded.seek(0)
# print('Downloaded file contents are: {}'.format(downloaded.read()))




# overalllist = []
# with open("images.txt", "r") as infile:
# 	image_list = json.load(infile)
# 	print("shape of image_list:")
# 	print(np.array(image_list).shape)
# 	overalllist.append(np.array(image_list))
# with open("types.txt", "r") as infile:
# 	value_list = json.load(infile)
# 	print("shape of value_list:")
# 	print(np.array(value_list).shape)
# 	overalllist.append(np.array(image_list))
# with open("imagestest.txt", "r") as infile:
# 	image_list_test = json.load(infile)
# 	print("shape of image_list_test:")
# 	print(np.array(image_list_test).shape)
# 	overalllist.append(np.array(image_list_test))
# with open("typestest.txt", "r") as infile:
# 	value_list_test = json.load(infile)
# 	print("shape of value_list_test:")
# 	print(np.array(value_list_test).shape)
# 	overalllist.append(np.array(value_list_test))

overalllist = imagetoarr.getimagearr()
train_X = overalllist[0]
train_Y = overalllist[1]
test_X = overalllist[2]
test_Y = overalllist[3]

print('Training data shape : ', train_X.shape, train_Y.shape)

print('Testing data shape : ', test_X.shape, test_Y.shape)

# Find the unique numbers from the train labels
classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))


# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))

#plt.show()

train_X = train_X.reshape(-1, 512,393, 1)
test_X = test_X.reshape(-1, 512,393, 1)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 20
num_classes = 3


# no dropout

# car_model = Sequential()
# car_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
# car_model.add(LeakyReLU(alpha=0.1))
# car_model.add(MaxPooling2D((2, 2),padding='same'))
# car_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
# car_model.add(LeakyReLU(alpha=0.1))
# car_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# car_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
# car_model.add(LeakyReLU(alpha=0.1))                  
# car_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# car_model.add(Flatten())
# car_model.add(Dense(128, activation='linear'))
# car_model.add(LeakyReLU(alpha=0.1))                  
# car_model.add(Dense(num_classes, activation='softmax'))
# car_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

# car_model.summary()

# fashion_train = car_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

# test_eval = car_model.evaluate(test_X, test_Y_one_hot, verbose=0)
# print('Test loss:', test_eval[0])
# print('Test accuracy:', test_eval[1])


#512 x 393
# with dropout

car_model = Sequential()
car_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(512,393,1)))
car_model.add(LeakyReLU(alpha=0.1))
car_model.add(MaxPooling2D((10, 10),padding='same'))
car_model.add(Dropout(0.25))
car_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
car_model.add(LeakyReLU(alpha=0.1))
car_model.add(MaxPooling2D(pool_size=(10, 10),padding='same'))
car_model.add(Dropout(0.25))
car_model.add(Flatten())
car_model.add(Dense(128, activation='linear'))
car_model.add(LeakyReLU(alpha=0.1))           
car_model.add(Dropout(0.3))
car_model.add(Dense(num_classes, activation='softmax'))
car_model.summary()

car_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
fashion_train_dropout = car_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

car_model.save("car_model_dropout.h5py")

test_eval = car_model.evaluate(test_X, test_Y_one_hot, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

accuracy = fashion_train_dropout.history['acc']
val_accuracy = fashion_train_dropout.history['val_acc']
loss = fashion_train_dropout.history['loss']
val_loss = fashion_train_dropout.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

predicted_classes = car_model.predict(test_X)

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
predicted_classes.shape, test_Y.shape

correct = np.where(predicted_classes==test_Y)[0]
#print "Found %d correct labels" + len(correct)
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[correct].reshape(512,393), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()

incorrect = np.where(predicted_classes!=test_Y)[0]
#print "Found %d incorrect labels" + len(incorrect)
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorrect].reshape(512,393), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
    plt.tight_layout()

from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))