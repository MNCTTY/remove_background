import imghdr
import os, glob
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import sys
from PIL import Image, ImageFilter
import cv2
import skimage.exposure
sys.path.append('u2net')
from u2net.prediction_class import Predict
import numpy as np
weights1 = 'u2net' #misc
weights2 = 'u2net_jewelry' #jewelry
weights3 = 'u2net_human' #people

prediction1 = Predict(weights1)
prediction2 = Predict(weights2)
prediction3 = Predict(weights3)


app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif', '.bmp', '.jpe', '.jfif', '.tiff', '.tif']
app.config['UPLOAD_PATH'] = 'uploads'


def validate_image(stream):
    # 512 bytes should be enough for a header check
    header = stream.read(512)
    # reset stream pointer
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')


@app.route('/')
def index():
    return render_template('index.html', files=[])


@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    
    img = cv2.imread(os.path.join(app.config['UPLOAD_PATH'], filename), cv2.IMREAD_ANYCOLOR)
    size = (img.shape[1], img.shape[0])
    
    result_files1 = prediction1.predict(os.path.join(app.config['UPLOAD_PATH'], filename))
    result_files2 = prediction2.predict(os.path.join(app.config['UPLOAD_PATH'], filename))
    result_files3 = prediction3.predict(os.path.join(app.config['UPLOAD_PATH'], filename))
    
    result_files = []
    result_files.append(os.path.join(app.config['UPLOAD_PATH'], f'{weights1}{os.path.splitext(filename)[0]}.png'))
    result_files.append(os.path.join(app.config['UPLOAD_PATH'], f'{weights2}{os.path.splitext(filename)[0]}.png'))
    result_files.append(os.path.join(app.config['UPLOAD_PATH'], f'{weights3}{os.path.splitext(filename)[0]}.png'))
    print(result_files)
    for f in result_files:
        if f.endswith('.png'):
            matting = cv2.imread(f, cv2.IMREAD_ANYCOLOR)
            
            th, im_th = cv2.threshold(matting, 150, 255, cv2.THRESH_BINARY)
            cv2.imwrite(f'{os.path.splitext(f)[0]}_binarized.png', im_th)
            
            binarized = cv2.imread(f'{os.path.splitext(f)[0]}_binarized.png')
            
            kernel = np.ones((5,5), np.uint8)
            #binarized_smoothed = cv2.erode(binarized,kernel,iterations = 1)
            binarized_smoothed = binarized
            cv2.imwrite(f'{os.path.splitext(f)[0]}_binarized_eroded.png', binarized_smoothed) 
            binarized_eroded = cv2.imread(f'{os.path.splitext(f)[0]}_binarized_eroded.png', cv2.IMREAD_GRAYSCALE)
            
            mask_ = binarized_eroded
            
            blurred_mask = cv2.GaussianBlur(mask_, (21, 21), 0)
            mask_of_mask = np.zeros(mask_.shape, np.uint8)

#             gray = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
#             cv2.imwrite(f'{os.path.splitext(f)[0]}_gray.png', gray)

            thresh = cv2.threshold(mask_of_mask, 60, 255, cv2.THRESH_BINARY)[1]
            contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            res = cv2.drawContours(mask_of_mask, contours, -1, (255,255,255),5)
            cv2.imwrite(f'{os.path.splitext(f)[0]}_res.png', res)

            output = np.where(mask_of_mask==np.array([255, 255, 255]), blurred_mask, mask_)
            cv2.imwrite(f'{os.path.splitext(f)[0]}_output.png', output)

            result_mask = cv2.imread(f'{os.path.splitext(f)[0]}_output.png', cv2.IMREAD_GRAYSCALE)
            
#             удаление фона как таковое

            result = cv2.bitwise_and(img, img, mask=result_mask)
            result[result_mask==0] = [255,255,255]
            
            
            cv2.imwrite(f'{os.path.splitext(f)[0]}_final.png', result)
            
            os.remove(f'{os.path.splitext(f)[0]}_binarized.png')
            os.remove(f'{os.path.splitext(f)[0]}_binarized_eroded.png')
            os.remove(f'{os.path.splitext(f)[0]}_output.png')
            os.remove(f'{os.path.splitext(f)[0]}_res.png')
#             os.remove(f'{os.path.splitext(f)[0]}_gray.png')
    
    return render_template('index.html', files=[filename] + ['u2net_human'+f'{os.path.splitext(filename)[0]}_final.png'] + ['u2net_jewelry'+f'{os.path.splitext(filename)[0]}_final.png']+['u2net'+f'{os.path.splitext(filename)[0]}_final.png'])

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)


@app.route('/uploads/')
def download():
    path = "mask.png"
    return send_file(path, as_attachment=True)

