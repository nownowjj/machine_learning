print('파일이 복사중입니다... 잠시만 기다려 주세요')

# 원본 이미지가 있는 폴더
origin_dada_folder = '../datasets/casts_and_dogs'

# 학습용  검증용 테스트용을 위한 개별 폴더 생성하기
target_folder = '../datasets/cats_and_dogs_small'

import os, shutil
if os.path.exists(target_folder):
    shutil.rmtree(target_folder)

os.mkdir(target_folder)

train_folder = os.path.join(target_folder ,' train')
os.mkdir(train_folder)

validation_folder = os.path.join(target_folder ,' validation')
os.mkdir(validation_folder)

test_folder = os.path.join(target_folder,'test')
os.mkdir(test_folder)

train_cats_folder = os.path.join(train_folder,'cats')
os.mkdir(train_cats_folder)

train_dogs_folder = os.path.join(train_folder,'dogs')
os.mkdir(train_dogs_folder)

validation_cats_folder = os.path.join(validation_folder,'cats')
os.mkdir(validation_cats_folder)

validation_dogs_folder = os.path.join(validation_folder,'dogs')
os.mkdir(validation_dogs_folder)

test_cats_folder = os.path.join(test_folder,'cats')
os.mkdir(test_cats_folder)

test_dogs_folder = os.path.join(test_folder,'dogs')
os.mkdir(test_dogs_folder)

# 고양이
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(origin_dada_folder,fname)
    dst = os.path.join(train_cats_folder,fname)
    shutil.copyfile(src,dst)

print('파일 갯수 확인')
print('훈련용 고양이 이미지 개수 : ' , len(os.listdir(train_cats_folder)))

print('finished')


