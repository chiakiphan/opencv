import cv2
import os
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import time
import matplotlib.pyplot as plt

google_link = "https://www.google.co.in/search?q={}&source=lnms&tbm=isch"
header = {
    'User-Agent': "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"}
img_link = '/home/kpst/PycharmProjects/face_classify_torch/face_detection_opencv/face_matching/image'
# img_link = '/home/kpst/PycharmProjects/face_classify_torch/face_detection_opencv/face_matching/video/train'
face_link = '/home/kpst/PycharmProjects/face_classify_torch/face_detection_opencv/face_matching/image/face'
video_link = '/home/kpst/Downloads/test_video.mp4'
raw_link = '/home/kpst/PycharmProjects/face_classify_torch/face_detection_opencv/face_matching/video/raw'
train_video_link = '/home/kpst/Downloads/videoplayback.mp4'
casade = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
label_name = {0: 'lee yong suk', 1: 'lee bo young'}
detect = cv2.CascadeClassifier(casade)
VIDEO_SIZE = (640, 480)
IMAGE_SIZE = (160, 160)
save_model = '/home/kpst/PycharmProjects/face_classify_torch/face_detection_opencv/face_matching/model.yml'


def save(url, img):
    with open(url, 'wb+') as writer:
        writer.write(img)


def download_image(name, number_image=40):
    link = google_link.format('+'.join(name.split()))
    sv_link = os.path.join(img_link, name)
    index = 0
    if os.path.exists(sv_link):
        return sv_link
    os.mkdir(sv_link)
    print('Download {} image....'.format(name))
    html = BeautifulSoup(requests.get(url=link, headers=header).content, 'html.parser')
    for src in html.findAll('img', {'class': 'rg_i Q4LuWd'}):
        if 'data-src' in src.attrs:
            img = requests.get(src.attrs['data-src']).content
            url = os.path.join(sv_link, name + str(index) + '.png')
            save(url, img)
            index += 1
        if number_image == 40:
            print('Done')
            break


def face_detect(image, minSize=None):
    gray = image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=minSize)
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        yield face, (x, y, w, h)


def get_face_video(video):
    cap = cv2.VideoCapture(video)
    index = 0
    print('Getting....')
    while cap.isOpened():
        ret, frame = cap.read()
        for face, _ in face_detect(frame, (80, 80)):
            index += 1
            cv2.imwrite(os.path.join(raw_link, 'vid{}.png'.format(index)), face)
            frame = draw_bbox(frame, _, color=[255, 0, 0])
        cv2.imshow('train', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    print('Done')


def get_face(file_name):
    url_save = None
    print('Getting....')
    for url in sorted(os.listdir(file_name)):
        if url_save is None:
            url_save = os.path.join(face_link, re.split(r'\d+', url)[0])
            if os.path.exists(url_save):
                return url_save
            os.mkdir(url_save)
        img = cv2.imread(os.path.join(file_name, url))
        for face, _ in face_detect(img):
            # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # plt.imshow(face)
            # plt.show()
            cv2.imwrite(os.path.join(url_save, url), face)
    print('Done')
    return url_save


def prepare_train_data(link):
    label = -1
    train, labels = [], []
    for person in link:
        label += 1
        for name in os.listdir(person):
            url = os.path.join(person, name)
            img = cv2.imread(url, 0)
            img = cv2.resize(img, IMAGE_SIZE)
            train.append(img)
            labels.append(label)
    return train, labels


def train_lbph(data):
    train, label = data
    # face_recog = cv2.face.LBPHFaceRecognizer_create()
    # face_recog = cv2.face.FisherFaceRecognizer_create()
    face_recog = cv2.face.EigenFaceRecognizer_create()
    face_recog.train(train, np.array(label))
    # face_recog.write(save_model)
    return face_recog


def predict_image(model):
    image = cv2.imread('/home/kpst/PycharmProjects/face_classify_torch/face_detection_opencv/image/img29.png', 0)
    image = cv2.resize(image, IMAGE_SIZE)
    # for face, _ in face_detect(image):
    label = model.predict(image)
    print(label)
    print(label_name[label[0]])


def draw_bbox(image, coord, color):
    x, y, w, h = coord
    image = cv2.rectangle(image,
                          (x, y),
                          (x + w, y + h),
                          color=color,
                          thickness=3)
    return image


def predict_video(model, link):
    cap = cv2.VideoCapture(link)
    color = [[255, 0, 0], [0, 0, 255]]
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, VIDEO_SIZE)
        frame_detect = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for face, coord in face_detect(frame_detect, minSize=(80, 80)):
            result = model.predict(cv2.resize(face, IMAGE_SIZE))
            label, conf = result
            if conf > 100:
                frame = draw_bbox(frame, coord, color[label])
        cv2.imshow('video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    # Crawl image
    lee_jong_suk = download_image('lee jong suk')
    lee_bo_young = download_image('lee bo young', number_image=80)
    # Get face from image
    # time.sleep(30)
    lee_jong_suk = get_face(lee_jong_suk)
    lee_bo_young = get_face(lee_bo_young)

    train, label = prepare_train_data((lee_jong_suk, lee_bo_young))
    model = train_lbph((train, label))
    # predict_image(model)
    predict_video(model, link=video_link)


if __name__ == '__main__':
    main()
    # get_face_video(train_video_link)
