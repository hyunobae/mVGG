import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.applications import resnet50


def dsetlist(flist, all_file):
    flist = []
    for f in all_file:
        img = Image.open(f)
        frame = np.array(img)
        flist.append(frame)
    return flist


# #load image and print brief info
path_true = ''
path_false = ''
# #advisible to use os.path.join as this makes concatenation OS independent
all_files_t = glob.glob(os.path.join(path_true, "*.png"))
# #advisible to use os.path.join as this makes concatenation OS independent
all_files_f = glob.glob(os.path.join(path_false, "*.png"))

# print(all_files_t, all_files_f)
flist_t = []
flist_t = dsetlist(flist_t, all_files_t)
flist_f = []
flist_f = dsetlist(flist_f, all_files_f)
print("# of images >>> T: ", len(flist_t), " F: ", len(flist_f))

l_x_train = []
l_y_train = []


def gen_ctu_dset(flist, l_x_train, l_y_train, atlas_class):
    """
    reshape keras-style array
    input
        @ flist: list of image files
        @ l_x_train, l_y_train: list for dataset
        @ atlas_class: class info. T/F
    """
    for f in range(len(flist)):
        # atlas_class = False
        frame = np.transpose(flist[f], (1, 0, 2))
        # init basic parameters
        width = len(frame)
        height = len(frame[0])
        channelnum = len(frame[0][0])

        ctusize = 128  # coding tree unit size
        nwctu = int(width / ctusize)
        nhctu = int(height / ctusize)
        numctu = nwctu * nhctu

        # crop beyond CTU boundary
        redframe = np.copy(frame[0:nwctu * ctusize, 0:nhctu * ctusize,:])
        print(redframe.shape, "--> CTU #: ", numctu)

        # arrange dataset for train : [CTU#, W, H, C]
        tmpnp = np.zeros((ctusize, ctusize, channelnum), dtype=np.int)
        for y in range(nhctu):
            for x in range(nwctu):
                posx = x * ctusize
                posy = y * ctusize
                # print("[{}]: {}, {}" .format(i, posx, posy))
                tmpnp = np.copy(redframe[posx:posx + ctusize, posy:posy + ctusize, :])
                l_x_train.append(tmpnp)

        if atlas_class == True:
            for y in range(nhctu):
                for x in range(nwctu):
                    l_y_train.append(np.ones(1, dtype=np.int))

        else:
            for y in range(nhctu):
                for x in range(nwctu):
                    l_y_train.append(np.zeros(1, dtype=np.int))
    return l_x_train, l_y_train


l_x_train, l_y_train = gen_ctu_dset(flist_t, l_x_train, l_y_train, True)
l_x_train, l_y_train = gen_ctu_dset(flist_f, l_x_train, l_y_train, False)

np_x_train = np.asarray(l_x_train)
np_y_train = np.asarray(l_y_train)

tmpy = np.array(np_y_train[:, 0])
print(tmpy.shape)
cnt_t = np.count_nonzero(tmpy == 1)
cnt_f = np.count_nonzero(tmpy == 0)
print("Trainset is ready: True of {}, False of {}".format(cnt_t, cnt_f))

exit()

seed = 0

np.random.seed(seed)
tf.random.set_seed(seed)

X_train, X_test, Y_train, Y_test = train_test_split(np_x_train, np_y_train,
                                                    test_size=0.3, random_state=seed)
ctusize = 128

X_train = X_train.reshape(X_train.shape[0], ctusize, ctusize, 3).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], ctusize, ctusize, 3).astype('float32') / 255
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

filepath = ''

k_size = (3, 3)
n_epoch = 200

history = []


#Model - VGGNet16
model = Sequential()
model.add(Conv2D(64, k_size, padding='same', input_shape=(ctusize, ctusize,3),
                strides=(1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, k_size, padding='same', strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(128, k_size, padding='same', strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, k_size, padding='same', strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(256, k_size, padding='same', strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, k_size, padding='same', strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, k_size, padding='same', strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(512, k_size, padding='same', strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, k_size, padding='same', strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, k_size, padding='same', strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(512, k_size, padding='same', strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, k_size, padding='same', strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, k_size, padding='same', strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(2, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model.summary()

modelpath = filepath + '/mVGG_{epoch:03d}-{val_acc:.4f}.hdf5'
checkpoint = ModelCheckpoint(modelpath, monitor='val_acc', verbose=0,
                             save_best_only=True, mode='max')

history.append(model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                         epochs=n_epoch, batch_size=32, verbose=2,
                         callbacks=[checkpoint]))

model.save(filepath + '/mVGG.h5')
del model

# Resnet Model

input = Input(shape=(128, 128, 3))
model = resnet50.ResNet50(weights= None, include_top=False, input_tensor= input, pooling='None')
x = model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs = model.input, outputs = predictions)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model.summary()

modelpath = filepath + '/resnet_{epoch:03d}-{val_acc:.4f}.hdf5'
checkpoint = ModelCheckpoint(modelpath, monitor='val_acc', verbose=0,
                             save_best_only=True, mode='max')

history.append(model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                         epochs=n_epoch, batch_size=32, verbose=2,
                         callbacks=[checkpoint]))

model.save(filepath + '/resnet.h5')
del model

print("\n")
y_vacc = []
y_acc = []
x_len = []
y_vloss = []
y_loss = []

plt.figure(figsize=(12, 7))

colorlist = ["red", "green"]
marklist = ['.', '+', 'o', '*', 'x']
for idx in range(2):
    y_vacc.append(history[idx].history['val_acc'])
    x_len.append(np.arange(len(y_vacc[idx])))
    plt.plot(x_len[idx], y_vacc[idx], marker=marklist[0], lw=1.0,
             c=colorlist[idx], label=str(idx))

plt.legend(('mVGG', 'Resnet'),
           shadow=True, loc='lower right')
ltext = plt.gca().get_legend().get_texts()
ftsize = 15
plt.setp(ltext[0], fontsize=ftsize)
plt.setp(ltext[1], fontsize=ftsize)

plt.grid()
plt.xlabel('Epoch')
plt.ylabel('V-Accuracy')
plt.show()
print("\n")

plt.figure(figsize=(12, 7))
for idx in range(2):
    y_acc.append(history[idx].history['acc'])
    x_len.append(np.arange(len(y_acc[idx])))
    plt.plot(x_len[idx], y_acc[idx], marker=marklist[0], lw=1.0,
             c=colorlist[idx], label=str(idx))
plt.legend(('mVGG', 'Resnet'),
           shadow=True, loc='lower right')
ltext = plt.gca().get_legend().get_texts()
ftsize = 15
plt.setp(ltext[0], fontsize=ftsize)
plt.setp(ltext[1], fontsize=ftsize)

plt.grid()
plt.xlabel('Epoch')
plt.ylabel('T-Accuracy')
plt.show()
print('\n')

# loss graph
plt.figure(figsize=(12, 7))

for idx in range(2):
    y_vloss.append(history[idx].history['val_loss'])
    x_len.append(np.arange(len(y_vloss[idx])))
    plt.plot(x_len[idx], y_vloss[idx], marker='.', lw=1.0, c=colorlist[idx],
             label=str(idx))

plt.legend(('mVGG', 'Resnet'),
           shadow=True, loc='upper right')
ltext = plt.gca().get_legend().get_texts()
ftsize = 15
plt.setp(ltext[0], fontsize=ftsize)
plt.setp(ltext[1], fontsize=ftsize)

plt.grid()
plt.xlabel('Epoch')
plt.ylabel('V-loss')
plt.show()
print('\n')