"""my_dataset_voc2012 dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds
import os

# TODO(my_dataset_voc2012): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(my_dataset_voc2012): BibTeX citation
_CITATION = """
"""

_NUM_SHARDS = 1

class MyDatasetVoc2012(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset_voc2012 dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(my_dataset_voc2012): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(shape=(None, None, 3)),
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
    # TODO(my_datasetVOC2012): Downloads the data and defines the splits
    paths: dict = {
        "images":       'D:\997_Datasets\VOC2012',
        "annotations":  'D:\997_Datasets\VOC2012',
        "labels":       'D:\997_Datasets\VOC2012',
    }

    images_path_dir = os.path.join(paths["images"], "JPEGImages")
    annotations_path_dir = os.path.join(paths["annotations"], "SegmentationClass")
    label_path_dir = os.path.join(paths["labels"], "ImageSets", "Segmentation")

    # Setup train and test splits
    train_split = tfds.core.SplitGenerator(
        name="train",
        # num_shards=_NUM_SHARDS,
        gen_kwargs={
            "images_dir_path": images_path_dir,
            "annotations_dir_path": annotations_path_dir,
            "images_list_file": os.path.join(label_path_dir, "train.txt"),
            },
        )
    test_split = tfds.core.SplitGenerator(
        name="test",
        # num_shards=_NUM_SHARDS,
        gen_kwargs={
            "images_dir_path": images_path_dir,
            "annotations_dir_path": annotations_path_dir,
            "images_list_file": os.path.join(label_path_dir, "val.txt")
            },
        )

    # return [train_split, test_split]
    return {
      'train': self._generate_examples(
          images_dir_path=images_path_dir,
          annotations_dir_path=annotations_path_dir,
          images_list_file=os.path.join(label_path_dir, "train.txt")
      ),
      'test': self._generate_examples(
          images_dir_path=images_path_dir,
          annotations_dir_path=annotations_path_dir,
          images_list_file=os.path.join(label_path_dir, "val.txt")
      ),
    }

  def _generate_examples(self, images_dir_path, annotations_dir_path, images_list_file):
    with tf.io.gfile.GFile(images_list_file, "r") as images_list:
      for line in images_list:
        # image_name, label, _, _ = line.strip().split(" ")
        image_name = line.replace( '\n' , '' )

        trimaps_dir_path = os.path.join(annotations_dir_path)

        trimap_name = image_name + ".png"
        image_name += ".jpg"
        label = int(1) - 1

        record = {
            "image": os.path.join(images_dir_path, image_name),
            "label": int(label),
            "file_name": image_name,
            "segmentation_mask": os.path.join(trimaps_dir_path, trimap_name)
        }

        yield image_name, record