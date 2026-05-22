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


def augment_crops(crops):
    """Apply random augmentation to each crop."""
    crops = tf.image.random_flip_left_right(crops)
    crops = tf.image.random_brightness(crops, max_delta=0.1)
    crops = tf.image.random_contrast(crops, lower=0.8, upper=1.2)
    crops = tf.clip_by_value(crops, 0.0, 1.0)
    return crops


def center_crop(image, crop_size=40):
    """Extract a single center crop from an image."""
    h, w = image.shape[0], image.shape[1]
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return image[top:top+crop_size, left:left+crop_size, :].astype(np.float32)


def get_dataloaders(path='datasets/9780a-main/FER-2013/fer2013.csv', bs=64, augment=True, use_ten_crop=True):
    fer2013, emotion_mapping = load_data(path)

    xtrain, ytrain = prepare_data(fer2013[fer2013['Usage'] == 'Training'])
    xval, yval = prepare_data(fer2013[fer2013['Usage'] == 'PrivateTest'])
    xtest, ytest = prepare_data(fer2013[fer2013['Usage'] == 'PublicTest'])

    mu, st = 0.0, 255.0
    xtrain = ((xtrain.astype(np.float32) - mu) / st)[..., np.newaxis]
    xval = ((xval.astype(np.float32) - mu) / st)[..., np.newaxis]
    xtest = ((xtest.astype(np.float32) - mu) / st)[..., np.newaxis]

    if use_ten_crop:
        xtrain_crops = np.array([ten_crop(img) for img in xtrain])
        xval_crops = np.array([ten_crop(img) for img in xval])
        xtest_crops = np.array([ten_crop(img) for img in xtest])
    else:
        xtrain_crops = np.array([center_crop(img) for img in xtrain])
        xval_crops = np.array([center_crop(img) for img in xval])
        xtest_crops = np.array([center_crop(img) for img in xtest])

    train_ds = tf.data.Dataset.from_tensor_slices(
        (xtrain_crops, ytrain.astype(np.int32)))
    if augment:
        train_ds = train_ds.cache().shuffle(len(xtrain)).map(
            lambda x, y: (augment_crops(x), y),
            num_parallel_calls=tf.data.AUTOTUNE).batch(bs).prefetch(tf.data.AUTOTUNE)
    else:
        train_ds = train_ds.cache().shuffle(len(xtrain)).batch(bs).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (xval_crops, yval.astype(np.int32)))
    val_ds = val_ds.cache().batch(bs).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (xtest_crops, ytest.astype(np.int32)))
    test_ds = test_ds.cache().batch(bs).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds
