# https://www.tensorflow.org/tutorials/images/segmentation

# t fds.load 404 error 解消方法
# GtiHub Issue - https://github.com/tensorflow/tensorflow/issues/31171
# xuthus commented on 9 Oct 2019 

# pip install tfds-nightly
# https://github.com/tensorflow/datasets/issues/2011

# コード内でデータセットをダウンロードすると、チェックサムエラーがでる
# Anaconda Promptで tfds build oxford_iiit_pet するとデータセットのDLができる

# 手動データセットの作成
# https://www.tensorflow.org/datasets/add_dataset#_split_generators_downloads_and_splits_data

# ポートレート画像に対するTrimap生成手法
# Trimapの説明
# https://db-event.jpn.org/deim2019/post/papers/183.pdf

# TensorFlowデータセットを簡単に操作するためのさまざまなコマンド
# https://www.tensorflow.org/datasets/cli


import tensorflow as tf
import tensorflow_datasets as tfds
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix
import os
import datasets.my_dataset
import datasets.my_dataset_voc2012

# dataset, info = tfds.load('my_dataset', with_info=True)
dataset, info = tfds.load('my_dataset_voc2012', with_info=True)
# dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True, download=True)

dl_paths: dict = {
  "images" :      'D:\890_gitfork\labelme\labelme\examples\semantic_segmentation\data_dataset_voc\JPEGImages',
  "annotations" :  "D:\890_gitfork\labelme\labelme\examples\semantic_segmentation\data_dataset_voc\SegmentationClassPNG",
  "list" :  "D:\890_gitfork\labelme\labelme\examples\semantic_segmentation\data_dataset_voc\SegmentationClassPNG\list.txt",
}

def gen_dataset(images_dir_path, annotations_dir_path, images_list_file):
  record_out: list = []
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

      record_out.append(record)
      # yield  image_name, record
  return record_out

dataset_: dict = {
  'train': gen_dataset(dl_paths["images"], dl_paths["annotations"], dl_paths["list"]),
  'test': gen_dataset(dl_paths["images"], dl_paths["annotations"], dl_paths["list"]),
}

# train: list = dataset['train']
# train_list: list = dataset_['train']
# train_ = dataset_['train']
# print(type(train), train_list, type(train_))

# tf.io.read_file(img_path)
print('checkpoint 000')

'''
画像は[0,1]に正規化されます。
セグメンテーションマスクのピクセルには{1、2、3}のいずれかのラベルが付けられます。
便宜上、セグメンテーションマスクから1を引いて、ラベルが{0、1、2}になるようにします。
'''
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  # input_mask -= 1 # <-- this line chaged
  input_mask /= 128 # <-- this line chaged
  return input_image, input_mask

'''
・128にリサイズ
・画像反転
・正規化
>>> tf.random.uniform(())
<tf.Tensor: shape=(), dtype=float32, numpy=0.10333085>
'''
@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


'''
・テストデータ
・128リサイズ
・正規化
'''
def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

###------------------------------------------------------------------------------------
###------------------------------------------------------------------------------------
###------------------------------------------------------------------------------------

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE) # <-- this line changed
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
# train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE) # <-- this line changed
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

'''
読み込んだデータを表示して確認する。
'''
def display(display_list):
  plt.figure(figsize=(15, 15))
  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

for image, mask in train.take(1):
  sample_image, sample_mask = image, mask

print(type(sample_mask), sample_mask[64,:])
display([sample_image, sample_mask])

OUTPUT_CHANNELS = 3

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

tf.keras.utils.plot_model(model, show_shapes=True)

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])

# 学習前に予測させてみる
show_predictions()

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

EPOCHS = 20 # <-- this line changed
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])



loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

show_predictions(test_dataset, 3)