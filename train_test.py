# Step 0: download and load the data
# load the dataset
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
import imgaug.augmenters as iaa
import imgaug as ia
from sklearn.model_selection import train_test_split
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
data_path = '/home/arash/data/Deep_FaceParts_Detection/'
Pface = np.moveaxis(np.load(data_path + 'face_images.npz')
                    ['face_images'], -1, 0)
LMs = pd.read_csv(data_path + 'facial_keypoints.csv')

LMpos = LMs.columns.tolist()
print(LMs.isnull().sum())
# I will only select the x and y of the eyes center,
# nose tip and mouth center, because these values are most available.
# This gives 7000 images and X and Y are build to fit Keras format.
iselect = np.nonzero(LMs.left_eye_center_x.notna() & LMs.right_eye_center_x.notna() &
                     LMs.nose_tip_x.notna() & LMs.mouth_center_bottom_lip_x.notna())[0]

Spic = Pface.shape[1]  # 96
m = iselect.shape[0]
X = np.zeros((m, Spic, Spic, 1))

# choose your face parts
num_values = 6
Y = np.zeros((m, num_values))
X[:, :, :, 0] = Pface[iselect, :, :]
Y[:, 0] = LMs.left_eye_center_x[iselect]
Y[:, 1] = LMs.left_eye_center_y[iselect]
Y[:, 2] = LMs.right_eye_center_x[iselect]
Y[:, 3] = LMs.right_eye_center_y[iselect]
Y[:, 4] = LMs.nose_tip_x[iselect]
Y[:, 5] = LMs.nose_tip_y[iselect]
# Y[:,6]=LMs.mouth_center_bottom_lip_x[iselect]
# Y[:,7]=LMs.mouth_center_bottom_lip_y[iselect]

print('# selected images = %d' % (m))

# Split the dataset
random_seed = 21
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=random_seed)


# Define a function to augment input and labels :
# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
def sometimes(aug): return iaa.Sometimes(0.5, aug)


# Define our sequence of augmentation steps that will be applied to every image
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        # iaa.Fliplr(0.5), # horizontally flip 50% of all images
        # iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            # scale images to 80-120% of their size, individually per axis
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate by -20 to +20 percent (per axis)
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),  # rotate by -45 to +45 degrees
            shear=(-16, 16),  # shear by -16 to +16 degrees
            # use nearest neighbour or bilinear interpolation (fast)
            order=[0, 1],
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            mode=ia.ALL
        ))
    ],
    random_order=True
)


def augment(batch_x, batch_y):

    # augmentation using imgaug library
    batch_x = batch_x.astype(np.uint8)
    batch_y = batch_y.astype(np.uint8)

    # get the y in imgaug kepionts format
    batch_y = batch_y.reshape((len(batch_x), -1, 2))
    keypoints = [[(x, y) for x, y in L] for L in batch_y]
    batch_x, points_aug = seq(images=batch_x, keypoints=keypoints)
    # get back in y format
    points_aug = [[list(elem) for elem in points] for points in points_aug]
    batch_y = np.array(points_aug).reshape((len(batch_x), -1))

    # normalize
    batch_x = batch_x.astype(np.float64) / 255
    batch_y = batch_y.astype(np.float64) / Spic

    return batch_x, batch_y

# Define the denerator that prepares the batch
def data_generator(X, Y, batch_size=64):
    while True:
        # Select files indices for the batch
        indxs = np.random.choice(a=range(len(X)),
                                 size=batch_size)
        batch_x = X[indxs]
        batch_y = Y[indxs]

        batch_x, batch_y = augment(batch_x, batch_y)

        yield batch_x, batch_y


# Define the model
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same',
                 activation='elu', input_shape=(Spic, Spic, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same', activation='elu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), padding='same', activation='elu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(num_values, activation='sigmoid'))

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='Adam')

# Training procedure
BS = 64
my_gen = data_generator(X_train, Y_train, batch_size=BS)
steps_per_epoch = len(X_train) // BS
model.fit_generator(my_gen, steps_per_epoch=steps_per_epoch,
                    epochs=100, verbose=1)

# Testing
X_test, Y_test = augment(X_test, Y_test)
Y_test_pred = model.predict(X_test)
print(np.mean((Y_test_pred - Y_test)**2))

# show some sample outputs
n = 0
nrows = 4
ncols = 4
irand = np.random.choice(Y_test.shape[0], nrows * ncols)
fig, ax = plt.subplots(nrows, ncols, sharex=True,
                       sharey=True, figsize=[ncols * 2, nrows * 2])
for row in range(nrows):
    for col in range(ncols):
        ax[row, col].imshow(X_test[irand[n], :, :, 0], cmap='gray')
        ax[row, col].scatter(Y_test[irand[n], 0::2] * Spic,
                             Y_test[irand[n], 1::2] * Spic, marker='X', c='r', s=20)
        ax[row, col].scatter(Y_test_pred[irand[n], 0::2] * Spic,
                             Y_test_pred[irand[n], 1::2] * Spic, marker='+', c='b', s=30)
        ax[row, col].set_xticks(())
        ax[row, col].set_yticks(())
        ax[row, col].set_title('image index = %d' % (irand[n]), fontsize=10)
        n += 1
plt.suptitle('x: Label; +: CNN', fontsize=16)
plt.show()
