import numpy as np
import tensorflow as tf

from data.dataset import load_data, prepare_data


def ten_crop(image, crop_size=40):
    """Generate 10 crops (5 positions + horizontal flips) from a single image."""
    h, w = image.shape[0], image.shape[1]
    crops = []
    positions = [
        (0, 0), (0, w - crop_size),
        (h - crop_size, 0), (h - crop_size, w - crop_size),
        ((h - crop_size) // 2, (w - crop_size) // 2)
    ]
    for top, left in positions:
        crop = image[top:top+crop_size, left:left+crop_size, :]
        crops.append(crop)
        crops.append(np.flip(crop, axis=1).copy())
    return np.stack(crops).astype(np.float32)


def get_dataloaders(path='datasets/fer2013/fer2013.csv', bs=64, augment=True):
    fer2013, emotion_mapping = load_data(path)

    xtrain, ytrain = prepare_data(fer2013[fer2013['Usage'] == 'Training'])
    xval, yval = prepare_data(fer2013[fer2013['Usage'] == 'PrivateTest'])
    xtest, ytest = prepare_data(fer2013[fer2013['Usage'] == 'PublicTest'])

    mu, st = 0.0, 255.0
    xtrain = ((xtrain.astype(np.float32) - mu) / st)[..., np.newaxis]
    xval = ((xval.astype(np.float32) - mu) / st)[..., np.newaxis]
    xtest = ((xtest.astype(np.float32) - mu) / st)[..., np.newaxis]

    xtrain_crops = np.array([ten_crop(img) for img in xtrain])
    xval_crops = np.array([ten_crop(img) for img in xval])
    xtest_crops = np.array([ten_crop(img) for img in xtest])

    train_ds = tf.data.Dataset.from_tensor_slices(
        (xtrain_crops, ytrain.astype(np.int32)))
    train_ds = train_ds.shuffle(len(xtrain)).batch(bs).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (xval_crops, yval.astype(np.int32)))
    val_ds = val_ds.batch(64).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (xtest_crops, ytest.astype(np.int32)))
    test_ds = test_ds.batch(64).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds
