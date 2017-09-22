import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import keras.preprocessing.image as kimage
import keras.utils as kutils


class ArtistDataset(object):
    """
    A simple class to manipulate the dataset.

    # Arguments
        path_to_images: The directory that holds all the image files.
        path_to_train_info (optional): The filepath to the csvfile that
            stores all the training data info. Defaults to not loading.
        train_info (optional): A pandas dataframe holding all the info
            for all the training samples that is indexed by image filename.
            Defaults to None.
        artist_encoder (optional): A Scikit-Learn LabelEncoder fo the
            artists. Defaults to constructing its own if train_info is
            or path_to_train_info is provided.
        files (optional): The list of files in the dataset. Defaults
            to all the images in path_to_images.
        img_size (optional): The size at which to load an image.
            Defaults to the original size of the image.
    """

    def __init__(self, path_to_images, path_to_train_info="",
                 train_info=None, artist_encoder=None, files=None,
                 img_size=None):
        self.path_to_images = path_to_images

        # Read all the images in our directories if not provided
        if files is None:
            files = [fname for fname in os.listdir(self.path_to_images) if os.path.splitext(
                os.path.basename(fname))[1] == '.jpg']

        self.files = np.array(files)
        # This gives us a quick way to go from file names to indicies
        self.file_to_ind = pd.Series(
            np.arange(len(self.files)), index=self.files)
        self.img_size = img_size

        # Load in the train info if not provided and we have a path. Keep only
        # the rows that we have images for.
        if train_info is None and path_to_train_info:
            train_info = pd.read_csv(
                path_to_train_info, index_col="filename").loc[self.files]

        self.train_info = train_info
        # An attribute that tells if the dataset has labels or not
        self.labels = self.train_info is not None
        # If we have labels, we can group by artist, so we know which
        # paintings were painted by the same artist
        if self.labels:
            self.artist_groups = self.train_info.groupby('artist')
            # This is used by _construct_dual_batch. We create it here once
            # and mutate to save memory and time.
            self._selection_mask = pd.Series(
                np.ones(len(self.train_info.index), dtpe=bool), index=self.train_info.index)

        # Construct a artist encoder if we have the train_info and no encoder
        if self.labels and artist_encoder is None:
            artist_encoder = preprocessing.LabelEncoder()
            artist_encoder.fit(self.train_info['artist'].values)

        self.artist_encoder = artist_encoder

        # Calculate the number of artists in our dataset
        if self.labels:
            self.num_artists = len(self.artist_encoder.classes_)
        else:
            self.num_artists = -1

    def __len__(self):
        """
        Returns the number of image files in the dataset.
        """
        return len(self.files)

    def load_img(self, img_name):
        """
        A method to load an image from PATH_TO_IMAGE/img_name.

        # Arguments
            img_name: The filename of the image to load. The path
                to the file is determined by the dataset's
                path_to_images attribute.

        # Returns
            A PIL Image Instance

        # Raises
            ImportError: if PIL is not available.
        """
        # Create the path to the image and load it.
        img_path = os.path.join(self.path_to_images, img_name)
        return kimage.load_img(img_path, target_size=self.img_size)

    def _construct_dual_batch(self, batch_indicies):
        """
        This protected method creates another set of batch fnames in which
        half share the same artist and half don't.
        """
        # Figure out which half will have the same artist and which will have
        # different artists
        batch_fnames = self.files[batch_indicies]
        artist_inds = np.permutation(len(batch_fnames))
        same_artist_inds = artist_inds[:len(batch_fnames) / 2]
        diff_artist_inds = artist_inds[len(batch_fnames) / 2:]
        # Get the artists of the entire batch
        batch_artists = self.train_info['artist'].loc[batch_fnames].values

        batch_fnames_b = np.empty_like(batch_fnames)
        # Get paintings by the same artists
        for i in same_artist_inds:
            artist = batch_artists[i]
            # Get the group of paintings by the artist and choose 1
            batch_fnames_b[i] = np.random.choice(
                self.artist_groups.get_group(artist).index.values)

        # Get the paintings by different artist
        for i in diff_artist_inds:
            artist = batch_artists[i]
            # Mask out all the paintings by the same artist and Randomly
            # choose one.
            self._selection_mask[self.artist_groups.get_group(
                artist).index.values] = False
            batch_fnames_b[i] = np.random.choice(
                self.train_info.index[self._selection_mask])
            # Set the selection mask back to True so we can reuse it
            self._selection_mask[self.artist_groups.get_group(
                artist).index.values] = True

        return self.file_to_ind[batch_fnames_b]

    def create_single_input_batch(self, batch_indicies):
        """
        A method for creating a batch input to a model from an array of
        indicies. This model should only take one input.

        # Arguments
            batch_indicies: Indicies into this dataset to load as a batch
                input for a model.

        # Returns
            The batch of loaded numpy images. If the image size is not defined,
            the batch is a list of images. Otherwise, it's a 4d numpy array.
            If labels are included in this dataset, return a tuple of
            model input and labels. Otherwise just return the model input.
        """
        # Get the batch of filenames
        batch_fnames = self.files[batch_indicies]
        # Initialize a numpy array if we have a defined img size
        # Otherwise, just use a list
        batch_x = np.empty((len(batch_indicies),) + self.img_size + (3,)
                           ) if self.img_size is not None else [0 for i in range(len(batch_indicies))]
        # Load all the images
        for i, fname in enumerate(batch_fnames):
            batch_x[i] = self.load_img(fname)
        # Load labels if we have any and return
        if self.labels:
            batch_artists = self.train_info['artist'].loc[batch_fnames].values
            batch_y = kutils.to_categorical(self.artist_encoder.transform(
                batch_artists), num_classes=self.num_artists)
            return batch_x, batch_y
        # Otherwise just return our image batch
        return batch_x

    def create_dual_input_batch(self, batch_indicies_a, batch_indicies_b):
        """
        A method for creating a batch input to a model from two arrays of
        indicies. This model should take two inputs.

        # Arguments
            batch_indicies_a: Indicies into this dataset to load as a left batch
                input for a model.
            batch_indicies_a: Indicies into this dataset to load as a right
                batch input for a model.

        # Returns
            The batch-pair of loaded numpy images. Return w.r.t. img_size
            follows same convention as create_single_input_batch. If labels
            are included in this dataset, will return a 1d array of 0's or 1's
            telling whether or not two corresponding images in each batch
            is by the same artist or not.
        """
        batch_a = self.create_single_input_batch(batch_indicies_a)
        batch_b = self.create_single_input_batch(batch_indicies_b)
        # Split into x,y if we have labels
        if self.labels:
            batch_x1, batch_y1 = batch_a
            batch_x2, batch_y2 = batch_b
            # Check if the two artists are the same or not for each image
            same_artist = np.all(batch_y1 == batch_y2, axis=1)
            return [batch_x1, batch_x2], same_artist
        # If we don't have labels, we just return the two batch of images
        else:
            batch_x1 = batch_a
            batch_x2 = batch_b
            return [batch_x1, batch_x2]

    def create_batch(self, *args):
        """
        An overloaded method that will create different kinds of batches
        depending on the number of inputs.
        """
        if len(args) == 1:
            return self.create_single_input_batch(*args)
        elif len(args) == 2:
            return self.create_dual_input_batch(*args)
        else:
            raise NotImplementedError(
                "create_batch only accepts 1 or 2 inputs, given %s" % len(args))

    def _get_split_indicies(self, split, shuffle, seed):
        split_ind = int(split * len(self))
        val_split = slice(split_ind)
        train_split = slice(split_ind, None)
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            indicies = np.random.permutation(len(self))
            train_split = indicies[train_split]
            val_split = indicies[val_split]
        return train_split, val_split

    def validation_split(self, split=0.2, shuffle=False, seed=None):
        """
        Splits the dataset into two smaller datasets based on the split

        # Arguments:
            split (optional): The fraction of the dataset to make validation.
                Defaults to 0.2.
            shuffle (optional): Whether or not to randomly sample the validation
                set and train set from the parent dataset. Defaults to False.
            seed (optional): A seed for the random number generator.

        # Returns
            A train dataset with (1-split) fraction of the data and a validation
            dataset with split fraction of the data
        """
        # Get the indicies that go into each split
        train_split, val_split = self._get_split_indicies(split, shuffle, seed)
        # Create the train and val datasets and return
        train_data = ArtistDataset(self.path_to_images,
                                   train_info=self.train_info,
                                   artist_encoder=self.artist_encoder,
                                   files=self.files[train_split],
                                   img_size=self.img_size)
        val_data = ArtistDataset(self.path_to_images,
                                 train_info=self.train_info,
                                 artist_encoder=self.artist_encoder,
                                 files=self.files[val_split],
                                 img_size=self.img_size)
        return train_data, val_data


class ArtistDatasetGenerator(object):
    """
    An abstract iterator to create batches for a model using an ArtistDataset.

    # Arguments
        dataset: The dataset to generate from
        batch_size: The number of samples in one batch
        shuffle (optional): Whether or not to shuffle the dataset before each epoch.
            Defaults to True.
        seed (optional): A seed for the random number generator.
    """

    def __init__(self, dataset, batch_size=None, shuffle=False, seed=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.index_array = np.arange(len(self.dataset))

        # Infer the steps_per_epoch
        self.steps_per_epoch = int((len(self.dataset) + self.batch_size - 1) /
                                   self.batch_size)
        self.batch_argument_generator = self.create_batch_argument_generator()

    def create_batch_argument_generator(self):
        """
        This is an iterator that generates the necessary arguments needed to
        create each batch from the index array.
        """
        raise NotImplementedError()

    def __next__(self):
        # Get the next set of batch indicies and create the batch
        return self.dataset.create_batch(*next(self.batch_argument_generator))

    def toggle_shuffle(self):
        """
        Toggles whether the generator shuffles the dataset on/off.
        """
        self.shuffle = not self.shuffle

    def restart(self):
        """
        Restarts the generation of batches from the beginning of the dataset.
        """
        self.batch_argument_generator = self.create_batch_argument_generator()


class SingleInputGen(ArtistDatasetGenerator):
    """
    Creates batches for models that take single inputs. Inherits
    ArtistDatasetGenerator.
    """

    def create_batch_argument_generator(self):
        # Set the seed if we have one
        if self.seed is not None:
            np.random.seed(self.seed)
        while True:
            # Shuffle if we need to
            if self.shuffle:
                np.random.shuffle(self.index_array)
            # Generate the batch_indicies
            for i in range(0, len(self.index_array), self.batch_size):
                yield (self.index_array[i:i + self.batch_size],)


class DoubleInputGen(ArtistDatasetGenerator):
    """
    Creates batches for models that take double inputs. Inherits
    ArtistDatasetGenerator.
    """

    def create_batch_argument_generator(self):
        # This batch generation is a bit more complicated since we want
        # to get a nice 50-50 split between paintings that are by the same
        # artist and those that are not.

        # Set the seed if we have one
        if self.seed is not None:
            np.random.seed(self.seed)
        while True:
            # Shuffle if we need to
            if self.shuffle:
                np.random.shuffle(self.index_array)

            # Generate the batch_indicies
            for i in range(0, len(self.index_array), self.batch_size):
                batch_indicies_a = self.index_array[i:i + self.batch_size]
                # This will give us the nice 50-50 split
                batch_indicies_b = self.dataset._construct_dual_batch(
                    batch_indicies_a)
                yield batch_indicies_a, batch_indicies_b


class TestDatasetGen(ArtistDatasetGenerator):
    """
    Creates batches for evaluating on the test datasets.

    # Arguments
        path_to_submission_info: The filepath to submission_info.csv
    """

    def __init__(self, dataset, path_to_submission_info, batch_size=None,
                 shuffle=False, seed=None):
        super(TestDatasetGen, self).__init__(
            dataset, batch_size, shuffle, seed)
        self.submission_info = pd.read_csv(index_col="index")
        # Reset the steps to be one pass through of the submission_info
        self.steps_per_epoch = (self.submission_info.shape[0]
                                + self.batch_size - 1) / self.batch_size

    def create_batch_argument_generator(self):
        """
        Generates batches according to the submission_info
        """
        while True:
            # Generate the batch_indicies
            for i in range(0, len(self.index_array), self.batch_size):
                batch_fnames_a = self.submission_info['img1'].loc[i:i +
                                                                  self.batch_size]
                batch_fnames_b = self.submission_info['img2'].loc[i:i +
                                                                  self.batch_size]
                yield self.dataset.file_to_ind[batch_fnames_a], self.dataset.file_to_ind[batch_fnames_b]
