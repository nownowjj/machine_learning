from keras.preprocessing.image import ImageDataGenerator

idg = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, \
                         shear_range=0.2, zoom_range=0.2, \
                         horizontal_flip=True, vertical_flip=True, \
                         fill_mode='nearest')

import os
img_source = '../image/'
filelist = os.listdir(img_source)
print(filelist)

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

target_w,target_h = 150,150
image_gen_su = 10

import matplotlib.pyplot as plt

for onefile in filelist:
    sample_image = img_source + onefile
    # print(sample_image)
    myimage = load_img(sample_image,target_size=(target_w,target_h))

    x = img_to_array(myimage)
    # flow 메소드에 적용하기 위하여 4차원으로 형상 변경합니다.
    x = x.reshape((1,) + x.shape)

    idx = 0
    for batch in idg.flow(x,batch_size=1):
        idx += 1

        plt.figure(num=idx)
        plt.axis('off')
        newimg = array_to_img(batch[0])
        plt.imshow(newimg)
        currfile = onefile.split('.')[0]
        filename = '../myimage/' + currfile + str(idx).zfill(3) + '.png'
        plt.savefig(filename)

        if idx % image_gen_su == 0:
            break
    #inner for
    print(onefile + '작업이 완료되었습니다.')
#outer for

print('finished')

