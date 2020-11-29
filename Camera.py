#coding: utf-8

import cv2
import numpy as np 
import io
import threading
from threading import Condition

class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()
    pass

    def write(self, buf):
        # New frame, copy the existing buffer's content and notify all
        # clients it's available

        self.buffer.truncate()
        
        with self.condition:
            self.frame = self.buffer.getvalue()
            self.condition.notify_all()
        pass

        self.buffer.seek(0) 
            
        return self.buffer.write(buf)
    pass
pass

class Camera(object):
    
    def __init__(self):
        self.output = StreamingOutput()
        # 전체 프레임 카운트 
        self.frame_cnt = 0 
        
        self.video = cv2.VideoCapture(-1)
        self.video.set(cv2.CAP_PROP_FPS, 24)
        self.video.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'X264'))
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

        print( "Recording now ...." )

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

        print( "Recording is stopped." )
    pass

    def get_image(self):
        success, image = self.video.read()

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
        self.putTextLine( image, txt , x, y )

        return image
    pass

    def get_frame( self ) : 
        # get video frame

        img = self.get_image()
         
        _, jpg = cv2.imencode('.jpg', img) 
        
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

def generate_frame(camera): 
    output = camera.output
    while True:
        with output.condition:
            output.condition.wait()
            frame = output.frame
        pass

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    pass
pass

if __name__=='__main__':
    # web by flask framewwork
    from flask import Flask, render_template, Response, request, jsonify

    app = Flask(__name__, static_url_path='', static_folder='html/static', template_folder='html/templates')
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    camera = Camera() 
    camera.start_recording()

    @app.route( '/' )
    @app.route( '/index.html' )
    @app.route( '/index.htm' )
    def index(): 
        return render_template('index_camera_only.html')
    pass

    @app.route('/video_feed')
    def video_feed(): 
        return Response(generate_frame(camera), mimetype='multipart/x-mixed-replace; boundary=frame')
    pass 

    print( "## Normal WEB")
    app.run(host='0.0.0.0', port=80, threaded=True) 
pass