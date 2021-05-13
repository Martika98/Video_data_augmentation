# This is a sample Python script.
import cv2
import os
import vidaug.augmentors as va
from PIL import Image
import numpy as np
import random


def video2frames(directory):
    frames = []
    vidcap = cv2.VideoCapture(directory)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        try:
            image.shape
        except AttributeError:
            break
        else:
            image_rgb = image
                #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image_rgb)
        if cv2.waitKey(10) == 27:  # exit if Escape is hit
            break
    data = np.array(frames)
    # data = frames
    return data


def data_aug_vid2vid(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            data = video2frames(os.path.join(directory, filename))
            if data is None:
                continue
            aug, list_ = data_aug_vid(data)
            frames2video(aug, list_, filename)
            # video_aug.save("out.avi", save_all=True, append_images=video_aug[1:], duration=100, loop=0)
        else:
            continue


def data_aug_vid(data):
    image_list = []
    sometimes = lambda aug: va.Sometimes(1, aug)  # Used to apply augmentor with 100% probability
    seq = va.Sequential([  # randomly rotates the video with a degree randomly choosen from [-10, 10]
        sometimes(va.HorizontalFlip())  # horizontally flip the video with 100% probability
    ])
    video_aug = seq(data)
    for image_ in video_aug:
        img = Image.fromarray(image_, 'RGB')
        image_list.append(img)

    return video_aug, image_list


def frames2video(aug, filename, aug_type='_horizontal_flip'):
    height, width, _ = aug[0].shape
    base_name = os.path.splitext(filename)[0]
    dir_ = base_name + aug_type + '.avi'
    video = cv2.VideoWriter(dir_, 0, 20, (width, height))

    for image in aug:
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB )
        video.write(image)

    video.release()

def slight_colour_change(change, image_list):
    for image in image_list:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                a, b, c = image[i][j]
                if a + change < 255 < a - change:
                    image[i][j][0] = a
    return image_list

def give_random_background(path = r'C:\Users\Martina\Desktop\baza_danych_inz\back_ground'):
    dir_list = os.listdir(path)
    rand = random.randint(0, len(dir_list) - 1)
    iterator = 0
    for filename in dir_list:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            if rand == iterator:
                back_img = cv2.imread(os.path.join(path, filename))
                back_img = cv2.cvtColor(back_img, cv2.COLOR_BGR2BGRA)
                return back_img
            iterator = iterator +1
        else:
            continue
    return None


def background_colour_change(colour, image_list, image_mode = False):
    for i in range(image_list.shape[0]):
        hsv_image = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2HSV)
        alpha_image = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2BGRA)
        trans_mask = cv2.inRange(hsv_image, (5, 105, 10), (14, 170, 250))
        #((5, 105, 10), (14, 170, 250))tło przy pani z krótkimi włosami i grzywką na bok i grubszej pani o jadnej karnacji
        kernel = np.ones((7,7), np.uint8)
        trans_mask = cv2.morphologyEx(trans_mask, cv2.MORPH_OPEN, kernel)

        #kernel2 = np.ones((1, 1), np.uint8)
        #trans_mask = np.array(cv2.erode(trans_mask, kernel2, iterations=1), dtype=bool)

        trans_mask = np.array(trans_mask, dtype=bool)
        if image_mode:
            back_image = give_random_background()
            image2 = cv2.resize(back_image, (alpha_image.shape[1],alpha_image.shape[0]))
            trans_mask = np.logical_not(trans_mask)
            image2[trans_mask] = (0, 0, 0, 0)
        else:
            image2 = np.zeros((alpha_image.shape), np.uint8)
            image2[trans_mask] = (255,0,255,255)

        cv2.imwrite('dupa.jpg', image2)
        image2 = cv2.GaussianBlur(image2, (5, 5), 0)
        image2 = cv2.blur(image2, (3,3))
        print(image2.shape)

        image_list[i] = cv2.cvtColor(cv2.addWeighted(alpha_image, 1, image2, 0.8, 0.0), cv2.COLOR_BGRA2BGR)


        #trans_mask = np.array(trans_mask, dtype=bool)
        #image[trans_mask] = (255, 255, 255)
        #image = cv2.GaussianBlur(image[:,:,0], (1, 1), sigmaX=4, sigmaY=4, borderType=cv2.BORDER_DEFAULT)
        print(f"trans_mask pierwszego zdjęcia --> {trans_mask}")

        #test
        #test
    return image_list

def go_through(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            data = video2frames(os.path.join(directory, filename))
            if data is None:
                continue
            data_aug = background_colour_change(30, data) #zmiana koloru tła
            #data_aug = slight_colour_change(70, data) #zmiana przestrzeni barw?
            frames2video(data_aug, filename, 'purple_background')
            #video_aug.save("out.avi", save_all=True, append_images=video_aug[1:], duration=100, loop=0)
        else:
            continue

if __name__ == '__main__':
    directory = r'C:\Users\Martina\Desktop\baza_danych_inz'
    go_through(directory)
