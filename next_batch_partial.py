import numpy


def next_batch_partial(dataset, batch_size, partial_size,
                       fake_data=False, shuffle=True):
    """
    Return the next `batch_size` examples from the first partial_size
    elements of this data set.

    This is a copy-paste modification of the Dataset.next_batch method from
    `tensorflow/tensorflow/contrib/learn/python/learn/datasets/mnist.py <https://github.com/tensorflow/tensorflow/blob/ac8e67399d75edce6a9f94afaa2adb577035966e/tensorflow/contrib/learn/python/learn/datasets/mnist.py>`_.

    It depends on private variables of that class ... so is brittle
    """ # noqa
    def shuffle_if_requested():
        if shuffle:
            perm = numpy.arange(dataset._num_examples)
            numpy.random.shuffle(perm[0:partial_size])
            dataset._images = dataset.images[perm]
            dataset._labels = dataset.labels[perm]

    if fake_data:
        fake_image = [1] * 784
        if dataset.one_hot:
            fake_label = [1] + [0] * 9
        else:
            fake_label = 0
        return [fake_image for _ in range(batch_size)], [
                fake_label for _ in range(batch_size)
        ]
    start = dataset._index_in_epoch
    # Shuffle for the first epoch
    if dataset._epochs_completed == 0 and start == 0:
        shuffle_if_requested()
    # Go to the next epoch
    if start + batch_size > partial_size:
        # Finished epoch
        dataset._epochs_completed += 1
        # Get the rest examples in this epoch
        rest_num_examples = partial_size - start
        images_rest_part = dataset._images[start:partial_size]
        labels_rest_part = dataset._labels[start:partial_size]
        # Shuffle the data
        shuffle_if_requested()
        # Start next epoch
        start = 0
        dataset._index_in_epoch = batch_size - rest_num_examples
        end = dataset._index_in_epoch
        images_new_part = dataset._images[start:end]
        labels_new_part = dataset._labels[start:end]
        return (numpy.concatenate((images_rest_part, images_new_part), axis=0),
                numpy.concatenate((labels_rest_part, labels_new_part), axis=0))
    else:
        dataset._index_in_epoch += batch_size
        end = dataset._index_in_epoch
        return dataset._images[start:end], dataset._labels[start:end]
