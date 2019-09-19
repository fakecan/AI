from PIL import Image, ImageOps
import os, sys, glob

attractionList = ['cheomseongdae night', 'colosseum', 'damyang metasequoia road', 'seoul tower night', 'pyramid']
attractionFolderList = ['cheomseongdae', 'colosseum', 'damyang_metasequoia', 'n_seoul_tower','pyramid']

for attractionFolder in attractionFolderList:
    image_dir = './data_origin/' + attractionFolder + '/'
    # target_resize_dir = './data/' + attractionFolder + '/'
    target_resize_dir = './data/'
    # print(image_dir)  #   ./data_origin/cheomseongdae/
    # print(target_resize_dir) #   ./data/cheomseongdae/
    if not os.path.isdir(target_resize_dir):
        os.makedirs(target_resize_dir)
    # print(target_resize_dir) # ./data/cheomseongdae/
    files = glob.glob(image_dir + "*.*")
    # print(files) #   ['./data_origin/cheomseongdae\\000001.jpg', './data_origin/cheomseongdae\\000005.jpg', ..., ]
    # print(len(files)) #  174

    count = 1
    size = (224, 224)
    for file in files:
        # print(file) # ./data_origin/cheomseongdae\000001.jpg
        im = Image.open(file)
        im = im.convert('RGB')
        print("i: ", count, im.format, im.size, im.mode, file.split("/")[-1])   # i:  1 None (500, 333) RGB cheomseongdae\000001.jpg
        count += 1
        im = ImageOps.fit(im, size, Image.ANTIALIAS, 0, (0.5, 0.5))
        print(file.split("/")[-1])                  # cheomseongdae\000001.jpg
        print(file.split("/")[-1].split(".")[0])    # cheomseongdae\000001
        im.save(target_resize_dir + file.split("/")[-1].split(".")[0] + ".jpg", quality=100)
        #       ./data/cheomseongdae/ + cheomseongdae\000001 + "".jpg"



