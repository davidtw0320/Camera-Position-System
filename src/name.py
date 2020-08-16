#-*- coding: UTF-8 -*-
import os

path ="./raw/"    #指定需要读取文件的目录
files =os.listdir(path) #采用listdir来读取所有文件

for filename in files:
    print ("Loading: %s" %(filename))

    #Image = Image.open(os.path.join(DATA_DIR, filename), 'rb')
    path1 = "raw/"+filename
    print (path1)
    # 调整大小
    image  = Image.open(path1)
    new_image = image.resize((300,400))

    half_the_width = new_image.size[0] / 2
    half_the_height = new_image.size[1] / 2
    new_image = new_image.crop(
        (
        half_the_width - 112,
        half_the_height - 112,
        half_the_width + 112,
        half_the_height + 112
        )
     )
    #new_image.save(filename)
    path2="out/"+filename
    new_image.save(path2)

