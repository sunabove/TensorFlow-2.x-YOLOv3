# coding: utf-8

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import logging, sys, io, time, cv2, threading, socketserver
import numpy as np, tensorflow as tf

from time import sleep
from threading import Condition
from http import server

from yolov3.utils import * 

PAGE="""\
<html>
<head>
<title>Raspberry Pi - YOLO v3 </title>
</head>
<body>
<center>
    <img src="stream.mjpg" />
</center>
</body>
</html>
"""

class StreamingOutput(object):

    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()
    pass

    def write(self, buf):
        self.buffer.truncate()
        
        with self.condition:
            self.frame = self.buffer.getvalue()
            self.condition.notify_all()
        pass

        self.buffer.seek(0) 
            
        return self.buffer.write(buf)
    pass
pass

class StreamingHandler(server.BaseHTTPRequestHandler):
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                output = camera.output
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    pass

                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
                pass
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()
        pass
    pass
pass

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True
pass

class Camera(object):

    def __init__(self, debug = 0 ):
        self.debug = debug

        self.output = StreamingOutput()
        # 전체 프레임 카운트 
        self.frame_cnt = 0 
        self.rotation = 0 
        
        video = cv2.VideoCapture(-1)        
        video.set(cv2.CAP_PROP_FPS, 24)
        video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'X264'))
        self.video = video        

        # load yolo model
        self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(video.get(cv2.CAP_PROP_FPS)) 
        self.times = []

        self.yolo = Load_Yolo_model()
    pass 
    
    def __del__(self):
        self.video.release()
    pass

    def start_recording(self) :
        x = threading.Thread(target=self.start_recording_impl, args=[] )
        x.start()
    pass

    def start_recording_impl(self) :
        self.recording = True

        while self.recording: 
            #print( "recording ...." , end="" )
            image = self.get_image()

            _, jpg = cv2.imencode('.jpg', image ) 
            self.output.write( jpg )
            #print( "Done." )
        pass
    pass

    def stop_recording(self):
        self.recording = False 
    pass

    def process_img(self, image) :
        t1 = time.time()
        
        if self.rotation :
            image = cv2.rotate( image, rotateCode = 1)
        pass

        debug = self.debug
        Yolo = self.yolo
        times = self.times        
        
        input_size=YOLO_INPUT_SIZE
        CLASSES=YOLO_COCO_CLASSES
        score_threshold=0.3
        iou_threshold=0.45
        rectangle_colors=''

        original_frame = image

        image_data = original_frame

        preprocess = 1
        if preprocess :
            image_data = image_preprocess(np.copy(image_data), [input_size, input_size]) 
        pass
        
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)
            pass
        pass
        
        t2 = time.time()
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')
        
        times.append(t2-t1)
        times = times[-20:]

        self.times = times 
        
        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        
        debug and print("Time: {:.2f}ms, {:.1f} FPS".format(ms, fps))

        frame = draw_bbox(image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)

        self.putTextLine( image, f"{fps:.1f} FPS", 10, 40 )
        
        return image
    pass

    def get_image(self):
        success, image = self.video.read()

        self.frame_cnt += 1 

        if not success :
            h = 480
            w= 640
            # black blank image
            image = np.zeros(shape=[h, w, 3], dtype=np.uint8)
            pass 
        pass

        image = self.process_img(image)

        x = 10   # text x position
        y = 20   # text y position
        h = 20   # line height

        txt = f"YOLO Object Detection [{self.frame_cnt}]"
        self.putTextLine( image, txt, x, y )

        return image
    pass

    def get_frame( self ) : 
        # get video frame

        image = self.get_image()

        _, jpg = cv2.imencode('.jpg', image) 
        
        return jpg.tobytes()
    pass

    def putTextLine(self, image, txt, x, y ) :
        # opencv 이미지에 텍스트를 그린다.
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.4  # font size(scale)
        ft = 1    # font thickness 

        bg_color = (255, 255, 255) # text background color
        fg_color = (255,   0,   0) # text foreground color

        cv2.putText(image, txt, (x, y), font, fs, bg_color, ft + 2, cv2.LINE_AA)
        cv2.putText(image, txt, (x, y), font, fs, fg_color, ft    , cv2.LINE_AA) 
    pass
pass

if __name__=='__main__':
    print( "Hello ..." )

    camera = Camera()

    if "-vf" in sys.argv : 
        #camera rotation (in degrees)
        camera.rotation = 90
        print( f"Camera roatation = {camera.rotation}" )
    pass

    camera.start_recording()
    try:
        address = ('', 80)
        server = StreamingServer(address, StreamingHandler)
        print( "Hello...\nServer is running now ....." )
        server.serve_forever()
    finally:
        camera.stop_recording()
    pass 

pass
