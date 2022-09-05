img_source = '../image/'
sample_image = img_source + 'cat.jpg'

from keras.preprocessing.image import load_img

target_w , target_h = 150,150
myimage = load_img(sample_image,target_size=(target_w,target_h))

from keras.preprocessing.image import img_to_array
x = img_to_array(myimage)
print('before x.shpae : ',x.shape)

print('flow 메소드가 4차원을 요구하므로 차원 변경')
x = x.reshape((1,) +x.shape)
print('after x.shape : ',x.shape)

from keras.preprocessing.image import ImageDataGenerator

idg=ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, \
                         shear_range=0.2, zoom_range=0.2, \
                         horizontal_flip=True, vertical_flip=True, \
                         fill_mode='nearest')

import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img

idx = 0 # 카운터 변수
image_gen_su = 20 # 생성할 이미지 갯수
for batch in idg.flow(x, batch_size=1):
    # print('type(batch) :', type(batch))
    # print(batch)
    idx += 1

    plt.figure(num=idx)
    plt.axis('off')
    newimg = array_to_img(batch[0])
    plt.imshow(newimg)

    filename = '../myimage/mycat' + str(idx).zfill(3) + '.png'
    plt.savefig(filename)

    if idx % image_gen_su == 0 :
        break
print('finished')


