import cv2
import glob

if __name__ == '__main__':
    total_imgs =  glob.glob('./data/ffhq_mask/*.jpg')
    for img_loc in total_imgs:
        img = cv2.imread(img_loc)
        img = cv2.resize(img, (256,256), img)
        cv2.imwrite('./data/256_mask/'+img_loc[-14:], img)