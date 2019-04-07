import os
import random
from multiprocessing import Pool
from captcha.image import ImageCaptcha
import binascii
import time
start = time.time()

FONTS = [
"./fonts/Aller_Rg.ttf",
"./fonts/Lato-Bold.ttf",
"./fonts/Lato-Heavy.ttf",
"./fonts/Lato-Medium.ttf",
"./fonts/Lato-Semibold.ttf",
"./fonts/OpenSans-Bold.ttf",
"./fonts/OpenSans-Semibold.ttf",
"./fonts/Oswald-DemiBold.ttf",
"./fonts/Oswald-Light.ttf",
"./fonts/Oswald-Regular.ttf",
"./fonts/RobotoCondensed-Bold.ttf",
"./fonts/RobotoCondensed-Regular.ttf",
"./fonts/Roboto-Regular.ttf",
]
FONT_SIZES = [31, 32, 33, 34, 35, 36]
CHAR_SETS = '0123456789QWERTYUIOPASDFGHJKLZXCVBNM'
    
MAX_CHAR_NUM = 7
MIN_CHAR_NUM = 5
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 35

image = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT,font_sizes=FONT_SIZES, fonts=FONTS)

def test_font(fonts):
    try:
        img = ImageCaptcha(width=100, height=50,font_sizes=[40], fonts=[fonts])
        img.write("1234", './test.png')
        return True
    except:
        return False 

def gen_captcha(data):
    fpath, label = data
    image.write(label, fpath) 

def create_data(gen_dir, nb_sample, nb_jobs=16):
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)
    
    data = []
    f = open("./data/captcha.csv", "w")
    f.write("ImageId,Label\n")
    for i in range(nb_sample):
        label = ''
        char_num = random.randint(MIN_CHAR_NUM, MAX_CHAR_NUM)
        for j in range(char_num):
            label += random.choice(CHAR_SETS)
        fname = binascii.hexlify(os.urandom(16)).decode('ascii') + '.png'
        data.append((os.path.join(gen_dir, fname), label))

        f.write("{},{}\n".format(fname, label))
    f.close()

    p = Pool(nb_jobs)
    p.map(gen_captcha, data)

create_data("./data/captcha", 50000, nb_jobs=16)
