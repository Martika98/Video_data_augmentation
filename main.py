# This is a sample Python script.
import cv2
import os
import vidaug.augmentors as va
from PIL import Image
import numpy as np


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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image)
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB )
        video.write(image)

    video.release()

def slight_colour_change(change, image_list):
    for image in image_list:
        for pixel in image:
            a, b, c = pixel
            if a + change < 255 < a - change:
                pixel[0] = a
    return image_list

def background_colour_change(colour, image_list):
    for image in image_list:
        #trans_mask = np.zeros(image.shape[0], image.shape[1])
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        trans_mask = cv2.inRange(hsv_image, (5, 105, 10), (14, 170, 250)) #((5, 105, 10), (14, 170, 250))tło przy pani z krótkimi włosami i grzywką na bok i grubszej pani o jadnej karnacji
        kernel = np.ones((7, 7), np.uint8)
        trans_mask = cv2.morphologyEx(trans_mask, cv2.MORPH_OPEN, kernel)
        #trans_mask = np.array(trans_mask, dtype = bool)
        kernel2 = np.ones((1, 1), np.uint8)
        trans_mask = np.array(cv2.dilate(trans_mask, kernel2, iterations=1), dtype=bool)
        image[trans_mask] = (255, 255, 255)
        print(f"trans_mask pierwszego zdjęcia --> {trans_mask}")
    return image_list

def go_through(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            data = video2frames(os.path.join(directory, filename))
            if data is None:
                continue
            data_aug = background_colour_change(30, data)
            frames2video(data_aug, filename, 'background_colour_white01_zamykanie')
            # video_aug.save("out.avi", save_all=True, append_images=video_aug[1:], duration=100, loop=0)
        else:
            continue

if __name__ == '__main__':
    directory = r'C:\Users\Martina\Desktop\baza_danych_inz'
    go_through(directory)
