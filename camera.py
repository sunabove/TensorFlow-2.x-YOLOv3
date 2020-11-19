# coding: utf-8

import logging
import io
import cv2
import numpy as np 
import threading
import socketserver
from threading import Condition
from http import server

PAGE="""\
<html>
<head>
<title>Raspberry Pi - YOLO v3 </title>
</head>
<body>
<center>
    <h3>Raspberry Pi - YOLO v3</h3>
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
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
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

class Cv2Camera(object):

    def __init__(self, resolution='640x480', framerate=24):
        self.video = cv2.VideoCapture(-1)
        self.video.set(cv2.CAP_PROP_FPS, framerate)
        self.video.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'X264'))
        # 전체 프레임 카운트 
        self.frame_cnt = 0 
    pass

    def __enter__(self) :
        pass
    pass

    def __exit__(self, exc_type, exc_value, tb):
        self.video.release()
    pass
    
    def __del__(self):
        self.video.release()
    pass

    def start_recording(self, output) :
        x = threading.Thread(target=self.start_recording_impl, args=[] )
    pass

    def start_recording_impl(self, output) :
        img = self.get_image()
    pass

    def get_image(self):
        success, img = self.video.read()

        self.frame_cnt += 1 

        if not success :
            h = 480
            w= 640
            # black blank image
            img = np.zeros(shape=[h, w, 3], dtype=np.uint8)
            pass 
        pass
        
        x = 10   # text x position
        y = 20   # text y position
        h = 20   # line height

        txt = f"Hello [{self.frame_cnt}]"
        self.putTextLine( img, txt , x, y )

        return img
    pass

    def get_frame( self ) : 
        # get video frame

        img = self.get_image()
         
        _, jpg = cv2.imencode('.jpg', img) 
        
        return jpg.tobytes()
    pass

    def putTextLine(self, img, txt, x, y ) :
        # opencv 이미지에 텍스트를 그린다.
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.4  # font size(scale)
        ft = 1    # font thickness 

        bg_color = (255, 255, 255) # text background color
        fg_color = (255,   0,   0) # text foreground color

        cv2.putText(img, txt, (x, y), font, fs, bg_color, ft + 2, cv2.LINE_AA)
        cv2.putText(img, txt, (x, y), font, fs, fg_color, ft    , cv2.LINE_AA) 
    pass
pass

if __name__=='__main__':

    with Cv2Camera(resolution='640x480', framerate=24) as camera:
        output = StreamingOutput()
        #Uncomment the next line to change your Pi's Camera rotation (in degrees)
        #camera.rotation = 90
        camera.start_recording(output)
        try:
            address = ('', 80)
            server = StreamingServer(address, StreamingHandler)
            server.serve_forever()
        finally:
            camera.stop_recording()
        pass
    pass

pass
