img_source = '../image/'
img_target = '../myimage/'

img_dog = img_source + 'mydog.png'
print('원본 이미지 : ', img_dog)

from keras.preprocessing.image import load_img

image32 = load_img(img_dog, target_size=(32, 32))
print(type(image32))

import matplotlib.pyplot as plt

plt.figure()
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.imshow(image32)
filename = img_target + 'dog32.png'
plt.savefig(filename)
print(filename + ' 파일 저장됨')


image64 = load_img(img_dog, target_size=(64, 64))
plt.figure()
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.imshow(image64)
filename = img_target + 'dog64.png'
plt.savefig(filename)
print(filename + ' 파일 저장됨')

image224 = load_img(img_dog, target_size=(224, 224))
plt.figure()
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.imshow(image224)
filename = img_target + 'dog224.png'
plt.savefig(filename)
print(filename + ' 파일 저장됨')


from keras.preprocessing.image import img_to_array, array_to_img

arr_dog_224 = img_to_array(image224)
print(type(arr_dog_224))
print(arr_dog_224.shape)

# 저해상도 이미지를 만들어 주는 함수

def drop_resolution(x, scale=3.0):
    # x : 이미지 정보를 담고 있는 numpy 배열
    size = (x.shape[0], x.shape[1])
    small_size = (int(size[0]/scale), int(size[1]/scale))

    img = array_to_img(x) # 배열을 이미지로 변환
    small_img = img.resize(small_size, 3)

    plt.imshow(small_img)
    filename = img_target + 'drop_res_image(' + str(scale) + ').png'
    plt.savefig(filename)
    print(filename + ' 파일 저장됨')
# end def drop_resolution()

drop_resolution(arr_dog_224)

drop_resolution(arr_dog_224, scale=10.0)














