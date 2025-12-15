import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms


#データセットを読み込む
ds_train=datasets.FashionMNIST(
    root='data',
    train=True,
    download=True
)

print(f'dataset size:{len(ds_train)}')

#インデックスを指定してデータを取り出す
#画像とクラス番号の組になってる
image,target=ds_train[0]

print(type(image))
print(target)


image=transforms.functional.to_image(image)
print(image.shape,image.dtype)

plt.imshow(image,cmap='gray_r',vmin=0,vmax=255)
plt.title(target)
plt.show()


