from glob import glob
from PIL import Image
import sys
import numpy as np
import os
import imageio
import cv2


class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res


    def load_data(self, domain, batch_size=1, is_testing=False):
        ##is_testingによって"trainA,trainB,testA,testB"を設定
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        #対象フォルダのファイルパスリストを取得
        path = glob('./data/%s/%s/*' % (self.dataset_name, data_type))
        #batch_size個ランダムにファイルパスを取り出す。
        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing: #訓練時
                img = cv2.resize(img, self.img_res)  # (256, 256, 3)->(128, 128, 3)

                if np.random.random() > 0.5:  # 0.0以上、1.0未満）の乱数を返す。
                    img = np.fliplr(img)  # 画像(ndarray)を左右反転: np.fliplr()
            else:  #テスト時
                img = cv2.resize(img, self.img_res) # (256, 256, 3)->(128, 128, 3)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.   # -1～1の範囲に正規化

        return imgs   # (batch_size, 128, 128, 3)
    
    #学習結果チェック用
    def load_sample_data(self, img_file):
      #path = glob(self.sample_data)
      #img = self.imread(img_file)
      
      img = cv2.resize(np.array(img_file).astype(np.uint8), self.img_res)
      print("img3333.shape=", img.shape)
      img = np.array(img)/127.5 - 1.   # -1～1の範囲に正規化
      img = np.expand_dims(img, 0)  # (128, 128, 3) -> (1, 128, 128, 3)
      print("img.shape4444=",img.shape)
      return img


    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path_A = glob('./data/%s/%sA/*' % (self.dataset_name, data_type))
        path_B = glob('./data/%s/%sB/*' % (self.dataset_name, data_type))


        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # モデルがすべてを参照できるように、各パスリストからn_batches * batch_sizeのサンプル
        # 両方のドメインからのサンプル
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = cv2.resize(img_A, self.img_res)
                img_B = cv2.resize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)  #ドメインAの画像を反転
                        img_B = np.fliplr(img_B)  #ドメインBの画像を反転

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.    #-1~1に正規化
            imgs_B = np.array(imgs_B)/127.5 - 1.    #-1~1に正規化

            yield imgs_A, imgs_B

    def imread(self, path):
        #return imageio.imread(path, pilmode="RGB").astype(np.uint8)
        return Image.open(path)
