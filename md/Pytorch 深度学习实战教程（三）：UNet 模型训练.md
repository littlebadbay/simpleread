> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [cuijiahua.com](https://cuijiahua.com/blog/2020/03/dl-16.html)

> 本文进行实战学习，针对医学图像分割任务，讲解了训练模型的三个步骤：数据加载、模型选择、算法选择。.

摘要

本文进行实战学习，针对医学图像分割任务，讲解了训练模型的三个步骤：数据加载、模型选择、算法选择。

![](https://cuijiahua.com/wp-content/uploads/2020/03/dl-16-tese.png)

一、前言
----

本文属于 Pytorch 深度学习语义分割系列教程。

该系列文章的内容有：

*   Pytorch 的基本使用
*   语义分割算法讲解

PS：文中出现的所有代码，均可在我的 github 上下载，欢迎 Follow、Star：[点击查看](https://github.com/Jack-Cherish/Deep-Learning/tree/master/Pytorch-Seg/lesson-2)

二、项目背景
------

深度学习算法，无非就是我们解决一个问题的方法。选择什么样的网络去训练，进行什么样的预处理，采用什么 Loss 和优化方法，都是根据具体的任务而定的。

所以，让我们先看一下今天的任务。

[![](https://cuijiahua.com/wp-content/uploads/2020/03/dl-16-1.gif)](https://cuijiahua.com/wp-content/uploads/2020/03/dl-16-1.gif)

没错，就是 UNet 论文中的经典任务：医学图像分割。

选择它作为今天的任务，就是因为简单，好上手。

简单描述一个这个任务：如动图所示，给一张细胞结构图，我们要把每个细胞互相分割开来。

这个训练数据只有 30 张，分辨率为 512x512，这些图片是果蝇的电镜图。

好了，任务介绍完毕，开始准备训练模型。

三、UNet 训练
---------

想要训练一个深度学习模型，可以简单分为三个步骤：

*   数据加载：数据怎么加载，标签怎么定义，用什么数据增强方法，都是这一步进行。
*   模型选择：模型我们已经准备好了，就是该系列上篇文章讲到的 UNet 网络。
*   算法选择：算法选择也就是我们选什么 loss ，用什么优化算法。

每个步骤说的比较笼统，我们结合今天的医学图像分割任务，展开说明。

### 1、数据加载

这一步，可以做很多事情，说白了，无非就是图片怎么加载，标签怎么定义，为了增加算法的鲁棒性或者增加数据集，可以做一些数据增强的操作。

既然是处理数据，那么我们先看下数据都是什么样的，再决定怎么处理。

数据已经备好，都在这里（Github）：[点击查看](https://github.com/Jack-Cherish/Deep-Learning/tree/master/Pytorch-Seg/lesson-2/data)

如果 Github 下载速度慢，可以使用**文末的百度链接**下载数据集。

数据分为训练集和测试集，各 30 张，训练集有标签，测试集没有标签。

数据加载要做哪些处理，是根据任务和数据集而决定的，对于我们的分割任务，不用做太多处理，但由于数据量很少，仅 30 张，我们可以使用一些数据增强方法，来扩大我们的数据集。

Pytorch 给我们提供了一个方法，方便我们加载数据，我们可以使用这个框架，去加载我们的数据。看下伪代码：

<table><tbody><tr><td data-settings="show"><p>1</p><p>2</p><p>3</p><p>4</p><p>5</p><p>6</p><p>7</p><p>8</p><p>9</p><p>10</p><p>11</p><p>12</p><p>13</p><p>14</p><p>15</p><p>16</p><p>17</p><p>18</p><p>19</p><p>20</p><p>21</p><p>22</p><p>23</p><p>24</p><p>25</p></td><td><p># ================================================================== #</p><p>#&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Input pipeline for custom dataset&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; #</p><p># ================================================================== #</p><p># You should build your custom dataset as below.</p><p>class CustomDataset(torch.utils.data.Dataset):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;def __init__(self):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# TODO</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 1. Initialize file paths or a list of file names.</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pass</p><p>&nbsp;&nbsp;&nbsp;&nbsp;def __getitem__(self, index):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# TODO</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 2. Preprocess the data (e.g. torchvision.Transform).</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 3. Return a data pair (e.g. image and label).</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pass</p><p>&nbsp;&nbsp;&nbsp;&nbsp;def __len__(self):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# You should change 0 to the total size of your dataset.</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return 0</p><p># You can then use the prebuilt data loader.</p><p>custom_dataset = CustomDataset()</p><p>train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; batch_size=64,</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; shuffle=True)</p></td></tr></tbody></table>

这是一个标准的模板，我们就使用这个模板，来加载数据，定义标签，以及进行数据增强。

创建一个 dataset.py 文件，编写代码如下：

<table><tbody><tr><td data-settings="show"><p>1</p><p>2</p><p>3</p><p>4</p><p>5</p><p>6</p><p>7</p><p>8</p><p>9</p><p>10</p><p>11</p><p>12</p><p>13</p><p>14</p><p>15</p><p>16</p><p>17</p><p>18</p><p>19</p><p>20</p><p>21</p><p>22</p><p>23</p><p>24</p><p>25</p><p>26</p><p>27</p><p>28</p><p>29</p><p>30</p><p>31</p><p>32</p><p>33</p><p>34</p><p>35</p><p>36</p><p>37</p><p>38</p><p>39</p><p>40</p><p>41</p><p>42</p><p>43</p><p>44</p><p>45</p><p>46</p><p>47</p><p>48</p><p>49</p><p>50</p><p>51</p><p>52</p><p>53</p><p>54</p></td><td><p>import torch</p><p>import cv2</p><p>import os</p><p>import glob</p><p>from torch.utils.data import Dataset</p><p>import random</p><p>class ISBI_Loader(Dataset):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;def __init__(self, data_path):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 初始化函数，读取所有 data_path 下的图片</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.data_path = data_path</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))</p><p>&nbsp;&nbsp;&nbsp;&nbsp;def augment(self, image, flipCode):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 使用 cv2.flip 进行数据增强，filpCode 为 1 水平翻转，0 垂直翻转，-1 水平 + 垂直翻转</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;flip = cv2.flip(image, flipCode)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return flip</p><p>&nbsp;&nbsp;&nbsp;&nbsp;def __getitem__(self, index):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 根据 index 读取图片</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image_path = self.imgs_path[index]</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 根据 image_path 生成 label_path</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;label_path = image_path.replace('image', 'label')</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 读取训练图片和标签图片</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image = cv2.imread(image_path)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;label = cv2.imread(label_path)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 将数据转为单通道的图片</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image = image.reshape(1, image.shape[0], image.shape[1])</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;label = label.reshape(1, label.shape[0], label.shape[1])</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 处理标签，将像素值为 255 的改为 1</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if label.max() &gt; 1:</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;label = label / 255</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 随机进行数据增强，为 2 时不做处理</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;flipCode = random.choice([-1, 0, 1, 2])</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if flipCode != 2:</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image = self.augment(image, flipCode)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;label = self.augment(label, flipCode)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return image, label</p><p>&nbsp;&nbsp;&nbsp;&nbsp;def __len__(self):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 返回训练集大小</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return len(self.imgs_path)</p><p>if __name__ == "__main__":</p><p>&nbsp;&nbsp;&nbsp;&nbsp;isbi_dataset = ISBI_Loader("data/train/")</p><p>&nbsp;&nbsp;&nbsp;&nbsp;print("数据个数：", len(isbi_dataset))</p><p>&nbsp;&nbsp;&nbsp;&nbsp;train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; batch_size=2,</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; shuffle=True)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;for image, label in train_loader:</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;print(image.shape)</p></td></tr></tbody></table>

运行代码，你可以看到如下结果：

[![](https://cuijiahua.com/wp-content/uploads/2020/03/dl-16-2-2.png)](https://cuijiahua.com/wp-content/uploads/2020/03/dl-16-2-2.png)

解释一下代码：

__init__函数是这个类的初始化函数，根据指定的图片路径，读取所有图片数据，存放到 self.imgs_path 列表中。

__len__函数可以返回数据的多少，这个类实例化后，通过 len() 函数调用。

__getitem__函数是数据获取函数，在这个函数里你可以写数据怎么读，怎么处理，并且可以一些数据预处理、数据增强都可以在这里进行。我这里的处理很简单，只是将图片读取，并处理成单通道图片。同时，因为 label 的图片像素点是 0 和 255，因此需要除以 255，变成 0 和 1。同时，随机进行了数据增强。

augment 函数是定义的数据增强函数，怎么处理都行，我这里只是进行了简单的旋转操作。

在这个类中，你不用进行一些打乱数据集的操作，也不用管怎么按照 batchsize 读取数据。因为实例化这个类后，我们可以用 torch.utils.data.DataLoader 方法指定 batchsize 的大小，决定是否打乱数据。

Pytorch 提供给给我们的 DataLoader 很强大，我们甚至可以指定使用多少个进程加载数据，数据是否加载到 CUDA 内存中等高级用法，本文不涉及，就不再展开讲解了。

### 2、模型选择

模型我们已经选择完了，就用上篇文章《[Pytorch 深度学习实战教程（二）：UNet 语义分割网络](https://cuijiahua.com/blog/2019/12/dl-15.html)》讲解的 UNet 网络结构。

但是我们需要对网络进行微调，完全按照论文的结构，模型输出的尺寸会稍微小于图片输入的尺寸，如果使用论文的网络结构需要在结果输出后，做一个 resize 操作。为了省去这一步，我们可以修改网络，使网络的输出尺寸正好等于图片的输入尺寸。

创建 unet_parts.py 文件，编写如下代码：

<table><tbody><tr><td data-settings="show"><p>1</p><p>2</p><p>3</p><p>4</p><p>5</p><p>6</p><p>7</p><p>8</p><p>9</p><p>10</p><p>11</p><p>12</p><p>13</p><p>14</p><p>15</p><p>16</p><p>17</p><p>18</p><p>19</p><p>20</p><p>21</p><p>22</p><p>23</p><p>24</p><p>25</p><p>26</p><p>27</p><p>28</p><p>29</p><p>30</p><p>31</p><p>32</p><p>33</p><p>34</p><p>35</p><p>36</p><p>37</p><p>38</p><p>39</p><p>40</p><p>41</p><p>42</p><p>43</p><p>44</p><p>45</p><p>46</p><p>47</p><p>48</p><p>49</p><p>50</p><p>51</p><p>52</p><p>53</p><p>54</p><p>55</p><p>56</p><p>57</p><p>58</p><p>59</p><p>60</p><p>61</p><p>62</p><p>63</p><p>64</p><p>65</p><p>66</p><p>67</p><p>68</p><p>69</p><p>70</p><p>71</p></td><td><p>"""Parts of the U-Net model"""</p><p>"""https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""</p><p>import torch</p><p>import torch.nn as nn</p><p>import torch.nn.functional as F</p><p>class DoubleConv(nn.Module):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;"""(convolution =&gt; [BN] =&gt; ReLU) * 2"""</p><p>&nbsp;&nbsp;&nbsp;&nbsp;def __init__(self, in_channels, out_channels):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;super().__init__()</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.double_conv = nn.Sequential(</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nn.BatchNorm2d(out_channels),</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nn.ReLU(inplace=True),</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nn.BatchNorm2d(out_channels),</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nn.ReLU(inplace=True)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;def forward(self, x):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return self.double_conv(x)</p><p>class Down(nn.Module):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;"""Downscaling with maxpool then double conv"""</p><p>&nbsp;&nbsp;&nbsp;&nbsp;def __init__(self, in_channels, out_channels):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;super().__init__()</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.maxpool_conv = nn.Sequential(</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nn.MaxPool2d(2),</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;DoubleConv(in_channels, out_channels)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;def forward(self, x):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return self.maxpool_conv(x)</p><p>class Up(nn.Module):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;"""Upscaling then double conv"""</p><p>&nbsp;&nbsp;&nbsp;&nbsp;def __init__(self, in_channels, out_channels, bilinear=True):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;super().__init__()</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# if bilinear, use the normal convolutions to reduce the number of channels</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if bilinear:</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;else:</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.conv = DoubleConv(in_channels, out_channels)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;def forward(self, x1, x2):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x1 = self.up(x1)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# input is CHW</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;diffY = torch.tensor([x2.size()[2] - x1.size()[2]])</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;diffX = torch.tensor([x2.size()[3] - x1.size()[3]])</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;diffY // 2, diffY - diffY // 2])</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x = torch.cat([x2, x1], dim=1)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return self.conv(x)</p><p>class OutConv(nn.Module):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;def __init__(self, in_channels, out_channels):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;super(OutConv, self).__init__()</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;def forward(self, x):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return self.conv(x)</p></td></tr></tbody></table>

创建 unet_model.py 文件，编写如下代码：

<table><tbody><tr><td data-settings="show"><p>1</p><p>2</p><p>3</p><p>4</p><p>5</p><p>6</p><p>7</p><p>8</p><p>9</p><p>10</p><p>11</p><p>12</p><p>13</p><p>14</p><p>15</p><p>16</p><p>17</p><p>18</p><p>19</p><p>20</p><p>21</p><p>22</p><p>23</p><p>24</p><p>25</p><p>26</p><p>27</p><p>28</p><p>29</p><p>30</p><p>31</p><p>32</p><p>33</p><p>34</p><p>35</p><p>36</p><p>37</p><p>38</p><p>39</p><p>40</p><p>41</p></td><td><p>"""Full assembly of the parts to form the complete network"""</p><p>"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""</p><p>import torch.nn.functional as F</p><p>from .unet_parts import *</p><p>class UNet(nn.Module):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;def __init__(self, n_channels, n_classes, bilinear=True):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;super(UNet, self).__init__()</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.n_channels = n_channels</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.n_classes = n_classes</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.bilinear = bilinear</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.inc = DoubleConv(n_channels, 64)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.down1 = Down(64, 128)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.down2 = Down(128, 256)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.down3 = Down(256, 512)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.down4 = Down(512, 512)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.up1 = Up(1024, 256, bilinear)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.up2 = Up(512, 128, bilinear)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.up3 = Up(256, 64, bilinear)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.up4 = Up(128, 64, bilinear)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.outc = OutConv(64, n_classes)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;def forward(self, x):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x1 = self.inc(x)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x2 = self.down1(x1)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x3 = self.down2(x2)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x4 = self.down3(x3)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x5 = self.down4(x4)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x = self.up1(x5, x4)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x = self.up2(x, x3)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x = self.up3(x, x2)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x = self.up4(x, x1)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logits = self.outc(x)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return logits</p><p>if __name__ == '__main__':</p><p>&nbsp;&nbsp;&nbsp;&nbsp;net = UNet(n_channels=3, n_classes=1)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;print(net)</p></td></tr></tbody></table>

这样调整过后，网络的输出尺寸就与图片的输入尺寸相同了。

### 3、算法选择

选择什么 Loss 很重要，Loss 选择的好坏，都会影响算法拟合数据的效果。

选择什么 Loss 也是根据任务而决定的。我们今天的任务，只需要分割出细胞边缘，也就是一个很简单的二分类任务，所以我们可以使用 BCEWithLogitsLoss。

啥是 BCEWithLogitsLoss？BCEWithLogitsLoss 是 Pytorch 提供的用来计算二分类交叉熵的函数。

它的公式是：

[![](https://cuijiahua.com/wp-content/uploads/2020/03/dl-16-3.png)](https://cuijiahua.com/wp-content/uploads/2020/03/dl-16-3.png)

看过我机器学习系列教程的朋友，对这个公式一定不陌生，它就是 Logistic 回归的损失函数。它利用的是 Sigmoid 函数阈值在 [0,1] 这个特性来进行分类的。

具体的公式推导，可以看我的机器学习系列教程《[机器学习实战教程（六）：Logistic 回归基础篇之梯度上升算法](https://cuijiahua.com/blog/2017/11/ml_6_logistic_1.html)》，这里就不再累述。

目标函数，也就是 Loss 确定好了，怎么去优化这个目标呢？

最简单的方法就是，我们耳熟能详的梯度下降算法，逐渐逼近局部的极值。

但是这种简单的优化算法，求解速度慢，也就是想找到最优解，费劲儿。

各种优化算法，本质上其实都是梯度下降，例如最常规的 SGD，就是基于梯度下降改进的随机梯度下降算法，Momentum 就是引入了动量的 SGD，以指数衰减的形式累计历史梯度。

除了这些最基本的优化算法，还有自适应参数的优化算法。这类算法最大的特点就是，每个参数有不同的学习率，在整个学习过程中自动适应这些学习率，从而达到更好的收敛效果。

本文就是选择了一种自适应的优化算法 RMSProp。

由于篇幅有限，这里就不再扩展，讲解这个优化算法单写一篇都不够，要弄懂 RMSProp，你得先知道什么是 AdaGrad，因为 RMSProp 是基于 AdaGrad 的改进。

比 RMSProp 更高级的优化算法也有，比如大名鼎鼎的 Adam，它可以看做是修正后的 Momentum+RMSProp 算法。

总之，对于初学者，你只要知道 RMSProp 是一种自适应的优化算法，比较高级就行了。

下面，我们就可以开始写训练 UNet 的代码了，创建 train.py 编写如下代码：

<table><tbody><tr><td data-settings="show"><p>1</p><p>2</p><p>3</p><p>4</p><p>5</p><p>6</p><p>7</p><p>8</p><p>9</p><p>10</p><p>11</p><p>12</p><p>13</p><p>14</p><p>15</p><p>16</p><p>17</p><p>18</p><p>19</p><p>20</p><p>21</p><p>22</p><p>23</p><p>24</p><p>25</p><p>26</p><p>27</p><p>28</p><p>29</p><p>30</p><p>31</p><p>32</p><p>33</p><p>34</p><p>35</p><p>36</p><p>37</p><p>38</p><p>39</p><p>40</p><p>41</p><p>42</p><p>43</p><p>44</p><p>45</p><p>46</p><p>47</p><p>48</p><p>49</p><p>50</p><p>51</p></td><td><p>from model.unet_model import UNet</p><p>from utils.dataset import ISBI_Loader</p><p>from torch import optim</p><p>import torch.nn as nn</p><p>import torch</p><p>def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;# 加载训练集</p><p>&nbsp;&nbsp;&nbsp;&nbsp;isbi_dataset = ISBI_Loader(data_path)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; batch_size=batch_size,</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; shuffle=True)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;# 定义 RMSprop 算法</p><p>&nbsp;&nbsp;&nbsp;&nbsp;optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;# 定义 Loss 算法</p><p>&nbsp;&nbsp;&nbsp;&nbsp;criterion = nn.BCEWithLogitsLoss()</p><p>&nbsp;&nbsp;&nbsp;&nbsp;# best_loss 统计，初始化为正无穷</p><p>&nbsp;&nbsp;&nbsp;&nbsp;best_loss = float('inf')</p><p>&nbsp;&nbsp;&nbsp;&nbsp;# 训练 epochs 次</p><p>&nbsp;&nbsp;&nbsp;&nbsp;for epoch in range(epochs):</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 训练模式</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;net.train()</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 按照 batch_size 开始训练</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for image, label in train_loader:</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;optimizer.zero_grad()</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 将数据拷贝到 device 中</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image = image.to(device=device, dtype=torch.float32)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;label = label.to(device=device, dtype=torch.float32)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 使用网络参数，输出预测结果</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pred = net(image)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 计算 loss</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;loss = criterion(pred, label)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;print('Loss/train', loss.item())</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 保存 loss 值最小的网络参数</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if loss &lt; best_loss:</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;best_loss = loss</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;torch.save(net.state_dict(), 'best_model.pth')</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 更新参数</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;loss.backward()</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;optimizer.step()</p><p>if __name__ == "__main__":</p><p>&nbsp;&nbsp;&nbsp;&nbsp;# 选择设备，有 cuda 用 cuda，没有就用 cpu</p><p>&nbsp;&nbsp;&nbsp;&nbsp;device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')</p><p>&nbsp;&nbsp;&nbsp;&nbsp;# 加载网络，图片单通道 1，分类为 1。</p><p>&nbsp;&nbsp;&nbsp;&nbsp;net = UNet(n_channels=1, n_classes=1)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;# 将网络拷贝到 deivce 中</p><p>&nbsp;&nbsp;&nbsp;&nbsp;net.to(device=device)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;# 指定训练集地址，开始训练</p><p>&nbsp;&nbsp;&nbsp;&nbsp;data_path = "data/train/"</p><p>&nbsp;&nbsp;&nbsp;&nbsp;train_net(net, device, data_path)</p></td></tr></tbody></table>

为了让工程更加清晰简洁，我们创建一个 model 文件夹，里面放模型相关的代码，也就是我们的网络结构代码，unet_parts.py 和 unet_model.py。

创建一个 utils 文件夹，里面放工具相关的代码，比如数据加载工具 dataset.py。

这种模块化的管理，大大提高了代码的可维护性。

train.py 放在工程根目录即可，简单解释下代码。

由于数据就 30 张，我们就不分训练集和验证集了，我们保存训练集 loss 值最低的网络参数作为最佳模型参数。

如果都没有问题，你可以看到 loss 正在逐渐收敛。

[![](https://cuijiahua.com/wp-content/uploads/2020/03/dl-16-4.png)](https://cuijiahua.com/wp-content/uploads/2020/03/dl-16-4.png)

四、预测
----

模型训练好了，我们可以用它在测试集上看下效果。

在工程根目录创建 predict.py 文件，编写如下代码：

<table><tbody><tr><td data-settings="show"><p>1</p><p>2</p><p>3</p><p>4</p><p>5</p><p>6</p><p>7</p><p>8</p><p>9</p><p>10</p><p>11</p><p>12</p><p>13</p><p>14</p><p>15</p><p>16</p><p>17</p><p>18</p><p>19</p><p>20</p><p>21</p><p>22</p><p>23</p><p>24</p><p>25</p><p>26</p><p>27</p><p>28</p><p>29</p><p>30</p><p>31</p><p>32</p><p>33</p><p>34</p><p>35</p><p>36</p><p>37</p><p>38</p><p>39</p><p>40</p><p>41</p><p>42</p><p>43</p></td><td><p>import glob</p><p>import numpy as np</p><p>import torch</p><p>import os</p><p>import cv2</p><p>from model.unet_model import UNet</p><p>if __name__ == "__main__":</p><p>&nbsp;&nbsp;&nbsp;&nbsp;# 选择设备，有 cuda 用 cuda，没有就用 cpu</p><p>&nbsp;&nbsp;&nbsp;&nbsp;device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')</p><p>&nbsp;&nbsp;&nbsp;&nbsp;# 加载网络，图片单通道，分类为 1。</p><p>&nbsp;&nbsp;&nbsp;&nbsp;net = UNet(n_channels=1, n_classes=1)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;# 将网络拷贝到 deivce 中</p><p>&nbsp;&nbsp;&nbsp;&nbsp;net.to(device=device)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;# 加载模型参数</p><p>&nbsp;&nbsp;&nbsp;&nbsp;net.load_state_dict(torch.load('best_model.pth', map_location=device))</p><p>&nbsp;&nbsp;&nbsp;&nbsp;# 测试模式</p><p>&nbsp;&nbsp;&nbsp;&nbsp;net.eval()</p><p>&nbsp;&nbsp;&nbsp;&nbsp;# 读取所有图片路径</p><p>&nbsp;&nbsp;&nbsp;&nbsp;tests_path = glob.glob('data/test/*.png')</p><p>&nbsp;&nbsp;&nbsp;&nbsp;# 遍历所有图片</p><p>&nbsp;&nbsp;&nbsp;&nbsp;for test_path in tests_path:</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 保存结果地址</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;save_res_path = test_path.split('.')[0] + '_res.png'</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 读取图片</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;img = cv2.imread(test_path)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 转为灰度图</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 转为 batch 为 1，通道为 1，大小为 512*512 的数组</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;img = img.reshape(1, 1, img.shape[0], img.shape[1])</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 转为 tensor</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;img_tensor = torch.from_numpy(img)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 将 tensor 拷贝到 device 中，只用 cpu 就是拷贝到 cpu 中，用 cuda 就是拷贝到 cuda 中。</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;img_tensor = img_tensor.to(device=device, dtype=torch.float32)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 预测</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pred = net(img_tensor)</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 提取结果</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pred = np.array(pred.data.cpu()[0])[0]</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 处理结果</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pred[pred &gt;= 0.5] = 255</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pred[pred &lt; 0.5] = 0</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 保存图片</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cv2.imwrite(save_res_path, pred)</p></td></tr></tbody></table>

运行完后，你可以在 data/test 目录下，看到预测结果：

[![](https://cuijiahua.com/wp-content/uploads/2020/03/dl-16-5.png)](https://cuijiahua.com/wp-content/uploads/2020/03/dl-16-5.png)

大功告成！

五、最后
----

*   本文主要讲解了训练模型的三个步骤：数据加载、模型选择、算法选择。
*   这是一个简单的例子，训练正常的视觉任务，要复杂很多。比如：在训练模型的时候，需要根据模型在验证集上的准确率选择保存哪个模型；需要支持 tensorboard 方便我们观察 loss 收敛情况等等。

PS： 如果觉得本篇本章对您有所帮助，欢迎关注、评论、赞！

文中出现的所有代码，均可在我的 github 上下载，欢迎 Follow、Star：[点击查看](https://github.com/Jack-Cherish/Deep-Learning/tree/master/Pytorch-Seg/lesson-2)