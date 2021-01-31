'''
U-NetでPascal VOC 2012の画像をSemantic Segmentationする (TensorFlow)
https://qiita.com/tktktks10/items/0f551aea27d2f62ef708
'''

from PIL import Image
import glob
import os

# @staticmethod
def generate_paths(dir_original, dir_segmented):
    '''
       dir_original(str): 入力画像のディレクトリ
       dir_segmented(str): 教師画像のディレクトリ
    '''
    # ファイル名を取得
    paths_original = glob.glob(dir_original + "/*")
    paths_segmented = glob.glob(dir_segmented + "/*")

    if len(paths_original) == 0 or len(paths_segmented) == 0:
        raise FileNotFoundError("Could not load images.")
    # 教師画像の拡張子を.pngに書き換えたものが読み込むべき入力画像のファイル名になります
    filenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], paths_segmented))
    paths_original = list(map(lambda filename: dir_original + "/" + filename + ".jpg", filenames))

    return paths_original, paths_segmented

#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------

# paths_original, paths_segmented = generate_paths('D:\997_Datasets\VOC2012\JPEGImages', 'D:\997_Datasets\VOC2012\SegmentationClass')
paths_original, paths_segmented = generate_paths('D:\890_gitfork\labelme\labelme\examples\semantic_segmentation\data_dataset_voc\JPEGImages', 'D:\890_gitfork\labelme\labelme\examples\semantic_segmentation\data_dataset_voc\SegmentationClassPNG')

imageOriginal = Image.open(paths_original[0])       #パスから画像１枚をロード
imageSegmented = Image.open(paths_segmented[0])     #パスから画像１枚をロード
print('オリジナル画像タイプ:', imageOriginal.mode, '画像数', len(paths_original))
print('セグメンテーション画像タイプ', imageSegmented.mode, '画像数', len(paths_segmented)) #教師データを読み込んだ際，自動で"P"モード(パレットモード)になります