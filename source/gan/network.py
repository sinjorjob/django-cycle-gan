from __future__ import print_function, division
from glob import glob
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from gan.dataloader import DataLoader
from django.conf import settings
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import imageio
import cv2
import scipy
from PIL import Image


class CycleGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.model_dir = "./models"
        self.num_of_trials = 27
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'color_gray'
        # DataLoaderオブジェクトを使用して前処理されたデータセットをインポートする
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

        # D (PatchGAN)の出力shapeを計算
        patch = int(self.img_rows / 2**4)  # 128/16=8
        self.disc_patch = (patch, patch, 1)  # (8, 8, 1)

        self.gf = 32  # Gの最初の層のフィルタ数
        self.df = 64  # Dの最初の層のフィルタ数

        # Loss weights
        self.lambda_cycle = 10.0                    # サイクル一貫性損失の重み(どれだけサイクル一貫性損失を考慮するか？| 値が大きいほど元の画像と再構成した画像が可能な限り似たものになる)
        self.lambda_id = 0.9 * self.lambda_cycle    # 同一性損失の重み(この値を小さくすると不要な変化が起こりやすくなる）

        optimizer = Adam(0.0002, 0.5)
        
        # 識別器の構築とコンパイル
        self.d_A = self.build_discriminator()   # D_A
        self.d_B = self.build_discriminator()   # D_B
        self.d_A.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])
        self.d_B.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])
        
        #前回学習の重みロード
        self.d_A.load_weights('.\gan\models\D_A_param.hdf5')
        self.d_B.load_weights('.\gan\models\D_B_param.hdf5')

        #-------------------------
        # 生成器の計算グラフを構築
        #-------------------------

        # 生成器の構築
        self.g_AB = self.build_generator()  # G_AB
        self.g_BA = self.build_generator()  # G_BA

        # ドメインA、Bからの画像入力
        img_A = Input(shape=self.img_shape)    # (128, 128)
        img_B = Input(shape=self.img_shape)    # (12, 128)

        # 画像を他のドメインに翻訳
        fake_B = self.g_AB(img_A)    #生成器(G_AB)にドメインAの画像を入力してドメインBへ変換
        fake_A = self.g_BA(img_B)    #生成器(G_BA)にドメインBの画像を入力してドメインAへ変換
        # 元のドメインに再翻訳
        reconstr_A = self.g_BA(fake_B)  #生成器(G_BA)にA->Bに翻訳した画像を入れてドメインAに戻す
        reconstr_B = self.g_AB(fake_A)  #生成器(G_AB)にB->Aに翻訳した画像を入れてドメインBに戻す
        # 画像の恒等写像（＝同一性損失）
        img_A_id = self.g_BA(img_A)   # ドメインAの画像をAに変換する処理
        img_B_id = self.g_AB(img_B)   # ドメインBの画像をBに変換する処理

        # 複合モデルにして生成器のみを訓練するため識別器のパラメータを固定にする。
        self.d_A.trainable = False
        self.d_B.trainable = False

        # 翻訳した画像を識別器で判定
        valid_A = self.d_A(fake_A)   # G_BAでAに翻訳したものはD_Aで識別
        valid_B = self.d_B(fake_B)   # G_ABでBに翻訳したものはD_Bで識別

        # 複合モデルで生成器を訓練し、識別器をだませるようにする。(=識別器の損失が最大になるように生成器を更新)
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       img_A_id, img_B_id])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                              loss_weights=[1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              optimizer=optimizer)
        #前回学習の重みロード
        self.combined.load_weights('.\gan\models\combined_param.hdf5')


    @staticmethod
    def conv2d(layer_input, filters, f_size=4, normalization=True):
      """識別器"""
      d = Conv2D(filters, kernel_size=f_size,
                 strides=2, padding='same')(layer_input)
      d = LeakyReLU(alpha=0.2)(d)
      if normalization:
          d = InstanceNormalization()(d)
      return d
      

    @staticmethod
    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
          """"アップサンプリング中に使われる層"""
          u = UpSampling2D(size=2)(layer_input)
          u = Conv2D(filters, kernel_size=f_size, strides=1,
                     padding='same', activation='relu')(u)
          if dropout_rate:
              u = Dropout(dropout_rate)(u)
          u = InstanceNormalization()(u)
          u = Concatenate()([u, skip_input])
          return u


    def build_generator(self):
        """U-Net Generator"""
        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = self.conv2d(d0, self.gf)
        d2 = self.conv2d(d1, self.gf * 2)
        d3 = self.conv2d(d2, self.gf * 4)
        d4 = self.conv2d(d3, self.gf * 8)

        # Upsampling
        u1 = self.deconv2d(d4, d3, self.gf * 4)
        u2 = self.deconv2d(u1, d2, self.gf * 2)
        u3 = self.deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4,
                            strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)


    def build_discriminator(self):
      img = Input(shape=self.img_shape)

      d1 = self.conv2d(img, self.df, normalization=False)
      d2 = self.conv2d(d1, self.df * 2)
      d3 = self.conv2d(d2, self.df * 4)
      d4 = self.conv2d(d3, self.df * 8)

      validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

      return Model(img, validity)


    def sample_images(self, epoch, batch_i):
      r, c = 2, 3

      imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)   #corlo画像
      imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)   #白黒画像
        
      # Translate images to the other domain
      fake_B = self.g_AB.predict(imgs_A)    #corlo　-> 白黒
      fake_A = self.g_BA.predict(imgs_B)    #白黒 -> color
      # Translate back to original domain
      reconstr_A = self.g_BA.predict(fake_B)   #白黒 -> color
      reconstr_B = self.g_AB.predict(fake_A)   # color -> 白黒

      gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])   # color , 白黒, color, 白黒, color , 白黒

      # Rescale images 0 - 1
      gen_imgs = 0.5 * gen_imgs + 0.5

      titles = ['Original', 'Translated', 'Reconstructed']
      fig, axs = plt.subplots(r, c)
      cnt = 0
      for i in range(r):
          for j in range(c):
              axs[i,j].imshow(gen_imgs[cnt])
              axs[i, j].set_title(titles[j])
              axs[i,j].axis('off')
              cnt += 1
      fig.savefig("./data/result/%s/%d_%d.png" % (self.dataset_name, epoch + self.num_of_trials, batch_i))
      plt.show() 


    def train(self, epochs, batch_size=1, sample_interval=50):
      # Adversarial loss ground truths
      valid = np.ones((batch_size,) + self.disc_patch) #(64, 8, 8, 1)
      fake = np.zeros((batch_size,) + self.disc_patch) #(64, 8, 8, 1)

      for epoch in range(epochs):
          print("epoch=", epoch)
          for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

              # ----------------------
              #  Train Discriminators
              # ----------------------

              # Translate images to opposite domain
              fake_B = self.g_AB.predict(imgs_A)
              fake_A = self.g_BA.predict(imgs_B)

              # Train the discriminators (original images = real / translated = Fake)
              dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
              dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
              dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

              dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
              dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
              dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

              # Total discriminator loss
              d_loss = 0.5 * np.add(dA_loss, dB_loss)
              print("d_loss=", d_loss)

              # ------------------
              #  Train Generators
              # ------------------

              # Train the generators
              g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                    [valid, valid,
                                                     imgs_A, imgs_B,
                                                     imgs_A, imgs_B])
              # If at save interval => plot the generated image samples
              if batch_i % sample_interval == 0:
                  print("batch_i=",batch_i)
                  self.sample_images(epoch, batch_i)
                  #識別器(D_A)のモデル保存
                  D_A_model_file=os.path.join(self.model_dir, "D_A_param-" + str(epoch + self.num_of_trials) +  ".hdf5")
                  self.d_A.save_weights(D_A_model_file)
                  #識別器(D_B)のモデル保存
                  D_B_model_file=os.path.join(self.model_dir, "D_B_param-" + str(epoch + self.num_of_trials) +  ".hdf5")
                  self.d_B.save_weights(D_B_model_file)
                  #生成器(combined)のモデル保存
                  combined_model_file=os.path.join(self.model_dir, "combined_param-" + str(epoch + self.num_of_trials) +  ".hdf5")
                  self.combined.save_weights(combined_model_file)
                  print(str(epoch + self.num_of_trials),":モデルの保存完了")


    def predict(self, img_file):
        
      img = self.data_loader.load_sample_data(img_file)   # 画像をロード

      d_A_path = '.\gan\models\D_A_param.hdf5'
      d_B_path = '.\gan\models\D_B_param.hdf5'
      combined_path = '.\gan\models\combined_param.hdf5'
      self.d_A.load_weights(d_A_path)
      self.d_B.load_weights(d_B_path)
      self.combined.load_weights(combined_path)
      fake_A = self.g_BA.predict(img)  #白黒 -> color
      translated_img = self.g_AB.predict(fake_A)   # color -> 白黒
     

      return translated_img
    
