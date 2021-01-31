"""my_dataset dataset."""


import tensorflow as tf
import tensorflow_datasets as tfds
import glob
import os

# TODO(my_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(my_dataset): BibTeX citation
_CITATION = """
"""


_NUM_SHARDS = 1


class MyDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        # features=tfds.features.FeaturesDict({
        #     # These are the features of your dataset like images, labels ...
        #     'image': tfds.features.Image(shape=(500, 338, 3)),
        #     'label': tfds.features.ClassLabel(names=['no', 'yes']),
        # }),
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(),
            "label": tfds.features.ClassLabel(num_classes=37),
            "file_name": tfds.features.Text(),
            "segmentation_mask": tfds.features.Image(shape=(None, None, 1))
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(my_dataset): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    # path = dl_manager.download('http://localhost/data_dataset_voc/JPEGImages')
    # path = dl_manager.download_and_extract('http://localhost/data.zip')
    # path = './dummy_data/data_dataset_voc/JPEGImages'

    # data_path is a pathlib-like `Path('<manual_dir>/data.zip')`
    # Extract the manually downloaded `data.zip`
    # path = dl_manager.extract('./datasets/my_dataset/dammy_data/data.zip')
    # path = dl_manager.extract('./dammy_data/data.zip')    
    # path='D:\890_gitfork\labelme\labelme\examples\semantic_segmentation\data_dataset_voc\JPEGImages'
    # 'D:\890_gitfork\labelme\labelme\examples\semantic_segmentation\data_dataset_voc\SegmentationClassPNG'
    # TODO(my_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    
    dl_paths: dict = {
      "images" :      'D:\890_gitfork\labelme\labelme\examples\semantic_segmentation\data_dataset_voc\JPEGImages',
      "annotations" :  "D:\890_gitfork\labelme\labelme\examples\semantic_segmentation\data_dataset_voc\SegmentationClassPNG",
    }

    images_path_dir: str = os.path.join(dl_paths["images"])
    annotations_path_dir: str = os.path.join(dl_paths["annotations"])

    # Setup train and test splits
    train_split = tfds.core.SplitGenerator(
        name="train",
        # num_shards=_NUM_SHARDS,
        gen_kwargs={
            "images_dir_path": images_path_dir,
            "annotations_dir_path": annotations_path_dir,
            "images_list_file": os.path.join(annotations_path_dir, "list.txt"),
            },
        )
    test_split = tfds.core.SplitGenerator(
        name="test",
        # num_shards=_NUM_SHARDS,
        gen_kwargs={
            "images_dir_path": images_path_dir,
            "annotations_dir_path": annotations_path_dir,
            "images_list_file": os.path.join(annotations_path_dir, "list.txt")
            },
        )

    return [train_split, test_split]
    '''
    return {
        'train': self._generate_examples(path=path),
        'test': self._generate_examples(path=path),
    }
    '''

  def _generate_examples(self, images_dir_path, annotations_dir_path, images_list_file):
    with tf.io.gfile.GFile(images_list_file, "r") as images_list:
      for line in images_list:
        image_name, label, _, _ = line.strip().split(" ")

        trimap_name = image_name + ".png"
        image_name += ".jpg"
        label = int(label) - 1

        record = {
            "image": os.path.join(images_dir_path, image_name),
            "label": int(label),
            "file_name": image_name,
            "segmentation_mask": os.path.join(annotations_dir_path, trimap_name)
        }
        yield image_name, record

  '''
  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(my_dataset): Yields (key, example) tuples from the dataset
    for f in glob.glob(path+'/*.jpg'):
      yield f.name, {
          'image': f,
          'label': 'yes',
      }
  '''