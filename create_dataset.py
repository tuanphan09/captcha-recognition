import os
import random
from captcha.image import ImageCaptcha
import binascii

FONTS = [
    "./fonts/DroidSans.ttf",
    "./fonts/Sansation-Light.ttf",
    "./fonts/Sansation-Regular.ttf",
    "./fonts/SourceSansPro-Light.otf",
    "./fonts/SourceSansPro-Regular.otf",
    "./fonts/Titillium-Light.otf",
    "./fonts/Titillium-Regular.otf",

    "./fonts/AlexBrush-Regular.ttf",
    "./fonts/cac_champagne.ttf",
    "./fonts/DancingScript-Regular.otf",
    "./fonts/GrandHotel-Regular.otf",
    "./fonts/Allura-Regular.otf",
    "./fonts/GreatVibes-Regular.otf",
]
FONT_SIZES = [43, 42, 41, 40, 39, 38, 37, 36]
CHAR_SETS = '0123456789qwertyuiopasdfghjklzxcvbnm'
    
CHAR_NUM = 6
IMAGE_WIDTH = 120
IMAGE_HEIGHT = 50
def gen(gen_dir, nb_sample, fonts, font_sizes):
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)
    image = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT,font_sizes=font_sizes, fonts=fonts)
    f = open(gen_dir + ".csv", "w")
    f.write("ImageId,Label\n")
    for i in range(nb_sample):
        label = ''
        for j in range(CHAR_NUM):
            label += random.choice(CHAR_SETS)
        fname = binascii.hexlify(os.urandom(16)).decode('ascii') + '.png'
        image.write(label, os.path.join(gen_dir, fname)) 
        f.write("{},{}\n".format(fname, label))
        if i % 1000 == 0:
            print(str(int(1.0*i/nb_sample*100))+ "%")
    f.close()

gen("./data/train", 150000, FONTS[:-2], FONT_SIZES[:-2])
gen("./data/public_test", 50000, FONTS[:-2], FONT_SIZES[:-1])
gen("./data/testing", 100000, FONTS, FONT_SIZES)