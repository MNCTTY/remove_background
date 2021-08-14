import cv2
import os
import sys
from os.path import isfile, join
import numpy as np
from pathlib import Path
from multiproc import parallel_greenscreen
import bg
class Rmbg_video:
    # Инициализация папок
    def __init__(self):
        self.path_to_video = '/removebg/uploads/'
        self.frame_path = '/removebg/data/video_frames/frames/'
        self.fg_frame_path = '/removebg/data/video_frames/fg_frames/'
        self.result_frames_path = '/removebg/data/video_frames/result_frames/'
        self.result_save_path = '/removebg/uploads/'

    # Проверка/создание всех папок
    def check(self):
        Path(self.path_to_video).mkdir(parents=True, exist_ok=True)
        Path(self.frame_path).mkdir(parents=True, exist_ok=True)
        Path(self.fg_frame_path).mkdir(parents=True, exist_ok=True)
        Path(self.result_frames_path).mkdir(parents=True, exist_ok=True)
        Path(self.result_save_path).mkdir(parents=True, exist_ok=True)

    # Создаем видео из масок
    def make_matte_video(self,videoname):
        parallel_greenscreen(f'{self.path_to_video}{videoname}',worker_nodes=1,gpu_batchsize=4,model_name='u2net_human_seg' )
    # Режем оригинальное видео и видео из масок на фреймы
    def cut(self, videoname):
        os.system(f"ffmpeg -i {self.path_to_video}{videoname} -vf fps=30 {self.frame_path}frames_%03d.png")
        os.system(f"ffmpeg -i {self.path_to_video}{os.path.splitext(videoname)[0]}.matte.mp4 -vf fps=30 {self.fg_frame_path}frames_%03d.png")

    # Обрезаем каждый фрем оригинального видео по маскам
    def merge_frames(self):
        for frame in os.listdir(self.frame_path):
            frame_og = cv2.imread(os.path.join(self.frame_path, frame), cv2.IMREAD_ANYCOLOR)

            size = (frame_og.shape[1], frame_og.shape[0])

            frame_fg = cv2.imread(os.path.join(self.fg_frame_path,frame), cv2.IMREAD_GRAYSCALE)

            frame_fg_new = cv2.resize(frame_fg, size)

            th, im_th = cv2.threshold(frame_fg_new, 150, 255, cv2.THRESH_BINARY)
            cv2.imwrite(f'{os.path.splitext(frame)[0]}_binarized.png', im_th)

            binarized = cv2.imread(f'{os.path.splitext(frame)[0]}_binarized.png')

            kernel = np.ones((5, 5), np.uint8)
            # binarized_smoothed = cv2.erode(binarized,kernel,iterations = 1)
            binarized_smoothed = binarized
            cv2.imwrite(f'{os.path.splitext(frame)[0]}_binarized_eroded.png', binarized_smoothed)
            binarized_eroded = cv2.imread(f'{os.path.splitext(frame)[0]}_binarized_eroded.png', cv2.IMREAD_GRAYSCALE)

            mask_ = binarized_eroded

            blurred_mask = cv2.GaussianBlur(mask_, (21, 21), 0)
            mask_of_mask = np.zeros(mask_.shape, np.uint8)

            #             gray = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
            #             cv2.imwrite(f'{os.path.splitext(f)[0]}_gray.png', gray)

            thresh = cv2.threshold(mask_of_mask, 60, 255, cv2.THRESH_BINARY)[1]
            contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            res = cv2.drawContours(mask_of_mask, contours, -1, (255, 255, 255), 5)
            cv2.imwrite(f'{os.path.splitext(frame)[0]}_res.png', res)

            output = np.where(mask_of_mask == np.array([255, 255, 255]), blurred_mask, mask_)
            cv2.imwrite(f'{os.path.splitext(frame)[0]}_output.png', output)

            result_mask = cv2.imread(f'{os.path.splitext(frame)[0]}_output.png', cv2.IMREAD_GRAYSCALE)


            result = cv2.bitwise_and(frame_og, frame_og, mask=result_mask)
            result[result_mask==0] = [255,255,255]
            cv2.imwrite(f'{self.result_frames_path}{frame}', result)
            os.remove(f'{os.path.splitext(frame)[0]}_binarized.png')
            os.remove(f'{os.path.splitext(frame)[0]}_binarized_eroded.png')
            os.remove(f'{os.path.splitext(frame)[0]}_output.png')
            os.remove(f'{os.path.splitext(frame)[0]}_res.png')

    # Склеиваем полученные фреймы в видео
    def convert_to_video(self,videoname):
        os.system(f"ffmpeg -r 30 -i {self.result_frames_path}frames_%03d.png -vcodec libx264 -y {self.result_save_path}{os.path.splitext(videoname)[0]}_final.mp4")

    # Очищаем папки от старых фреймов
    def clear(self):
        folders = [self.frame_path, self.fg_frame_path, self.result_frames_path]
        for folder in folders:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))


    def process_video(self, videoname):
        self.check()
        self.make_matte_video(videoname)
        self.cut(videoname)
        self.merge_frames()
        self.convert_to_video(videoname)
        self.clear()

if __name__ == "__main__":
    videoname = 'video1.mp4'
    predict = Rmbg_video()
    predict.process_video(videoname)







