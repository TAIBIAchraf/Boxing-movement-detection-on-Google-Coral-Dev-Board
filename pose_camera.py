# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import threading
import argparse
import collections
from functools import partial
import time
import math
import numpy as np
import svgwrite
import gstreamer

from pose_engine import PoseEngine
from pose_engine import KeypointType


def switch(flag_id):
  if flag_id == 0:
    return ''
  elif flag_id == 1:
    return 'Droite en Haut'
  elif flag_id == 2:
    return 'Droite en Bas'
  elif flag_id == 3:
    return 'Gauche en Haut'
  elif flag_id == 4:
    return 'Droite en Bas'
  elif flag_id == 5:
    return 'Uppercut Droite'
  elif flag_id == 6:
    return 'Uppercut Gauche'



def vitess(xy_t0,xy_t1):
    # fps de l'inférence est en  ms left
    vitess_x = (xy_t1[0] - xy_t0[0]) / avg_inference_time
    vitess_y = (xy_t1[1] - xy_t0[1]) / avg_inference_time
    return math.sqrt(vitess_x ** 2 + vitess_y ** 2)

def getAngle(a_, b_, c_):
    a = np.array([a_[0], a_[1]])
    b = np.array([b_[0], b_[1]])
    c = np.array([c_[0], c_[1]])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)



# Pour dessiner les lignes après
EDGES = (
    #(KeypointType.NOSE, KeypointType.LEFT_EYE),
    #(KeypointType.NOSE, KeypointType.RIGHT_EYE),
    #(KeypointType.NOSE, KeypointType.LEFT_EAR),
    #(KeypointType.NOSE, KeypointType.RIGHT_EAR),
    #(KeypointType.LEFT_EAR, KeypointType.LEFT_EYE),
    #(KeypointType.RIGHT_EAR, KeypointType.RIGHT_EYE),
    #(KeypointType.LEFT_EYE, KeypointType.RIGHT_EYE),
    (KeypointType.LEFT_SHOULDER, KeypointType.RIGHT_SHOULDER),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_ELBOW),
    #(KeypointType.LEFT_SHOULDER, KeypointType.LEFT_HIP),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_ELBOW),
    #(KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_HIP),
    (KeypointType.LEFT_ELBOW, KeypointType.LEFT_WRIST),
    (KeypointType.RIGHT_ELBOW, KeypointType.RIGHT_WRIST),
    #(KeypointType.LEFT_HIP, KeypointType.RIGHT_HIP),
    #(KeypointType.LEFT_HIP, KeypointType.LEFT_KNEE),
    #(KeypointType.RIGHT_HIP, KeypointType.RIGHT_KNEE),
    #(KeypointType.LEFT_KNEE, KeypointType.LEFT_ANKLE),
    #(KeypointType.RIGHT_KNEE, KeypointType.RIGHT_ANKLE),
)

#Right and Left points pour deboggage
LEFTS_POINTS = (KeypointType.LEFT_SHOULDER,KeypointType.LEFT_ELBOW,KeypointType.LEFT_WRIST,
                KeypointType.LEFT_HIP,KeypointType.LEFT_KNEE,KeypointType.LEFT_EYE,KeypointType.LEFT_EAR)

RIGHTS_POINTS = (KeypointType.RIGHT_SHOULDER,KeypointType.RIGHT_ELBOW,KeypointType.RIGHT_WRIST,
                KeypointType.RIGHT_HIP,KeypointType.RIGHT_KNEE)


def shadow_text(dwg, x, y, text, font_size=16,color='white',flag = 0,iter_display = 0):
    if (flag == 0):
        dwg.add(dwg.text(text, insert=(x + 1, y + 1), fill='black',
                         font_size=font_size, style='font-family:sans-serif'))
        dwg.add(dwg.text(text, insert=(x, y), fill=color,
                         font_size=font_size, style='font-family:sans-serif'))
    elif(flag != 0 and iter_display <= 60):
            dwg.add(dwg.text(text, insert=(x + 1, y + 1), fill=color,
                             font_size=font_size, style='font-family:sans-serif'))
            dwg.add(dwg.text(text, insert=(x, y), fill=color,
                             font_size=font_size, style='font-family:sans-serif'))
    else:
        flag = 0


#variable pour l'itération des frames
iter_ = 0
#variable pour gérer la double détéction
db_coups = 0
#liste pour le comptage des coups (y en a 6)
counter_arr = [0,0,0,0,0,0,0]
#setup time en sec
setup_time_t0 = time.monotonic()
#init
delta_time_setup = 0
#exercice time set en s
exercice_time = 30
#flag pour incruster les coups
flag_incr = 0
#display counter
disp_counter = 0
#renvoi des stats
instruction_executee = False
def draw_pose(dwg, pose, src_size, inference_box, color='yellow', threshold=0.2):
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_size[0] / box_w, src_size[1] / box_h
    xys = {}

    global iter_,Right_wrist_x_y_t0,cpt_jab,Left_wrist_x_y_t0,db_coups,Left_shoulder_x_y_t0,Right_shoulder_x_y_t0,Right_elbow_x_y_t0,counter_arr\
            ,delta_time_setup, exercice_time,flag_incr,disp_counter
    #keypoint nose
    nose_x_y = pose.keypoints[KeypointType.NOSE][0]

    #keypoints récupérés (Droite)
    Right_shoulder_x_y = pose.keypoints[KeypointType.RIGHT_SHOULDER][0]
    Right_elbow_x_y =  pose.keypoints[KeypointType.RIGHT_ELBOW][0]
    Right_wrist_x_y = pose.keypoints[KeypointType.RIGHT_WRIST][0]

    #keypoints récupérés (Gauche)
    Left_shoulder_x_y = pose.keypoints[KeypointType.LEFT_SHOULDER][0]
    Left_elbow_x_y =  pose.keypoints[KeypointType.LEFT_ELBOW][0]
    Left_wrist_x_y = pose.keypoints[KeypointType.LEFT_WRIST][0]

    #cof de correlation si >0 alors gauche/droite haut sinon bas
    R_right = np.corrcoef([Right_shoulder_x_y[0],Right_elbow_x_y[0],Right_wrist_x_y[0]], [Right_shoulder_x_y[1],Right_elbow_x_y[1],Right_wrist_x_y[1]])
    R_left = np.corrcoef([Left_shoulder_x_y[0],Left_elbow_x_y[0],Left_wrist_x_y[0]], [Left_shoulder_x_y[1],Left_elbow_x_y[1],Left_wrist_x_y[1]])

    #setup time
    setup_time_t1 = time.monotonic()

    # Print le setup time
    delta_time_setup = setup_time_t1 - setup_time_t0

    if iter_ == 0:
        Right_wrist_x_y_t0 = pose.keypoints[KeypointType.RIGHT_WRIST][0]
        Left_wrist_x_y_t0 = pose.keypoints[KeypointType.LEFT_WRIST][0]

        Right_shoulder_x_y_t0 = pose.keypoints[KeypointType.RIGHT_SHOULDER][0]
        Left_shoulder_x_y_t0 = pose.keypoints[KeypointType.LEFT_SHOULDER][0]

        Right_elbow_x_y_t0 = pose.keypoints[KeypointType.RIGHT_ELBOW][0]
    elif iter_ != 0 and (int(delta_time_setup) >= 10) and (int(delta_time_setup) <= exercice_time + 10) :

        Right_elbow_x_y_t1 = pose.keypoints[KeypointType.RIGHT_ELBOW][0]

        #la vitesse / accélérationd'un point
        Right_wrist_x_y_t1 = pose.keypoints[KeypointType.RIGHT_WRIST][0]
        Left_wrist_x_y_t1 = pose.keypoints[KeypointType.LEFT_WRIST][0]

        Right_shoulder_x_y_t1 = pose.keypoints[KeypointType.RIGHT_SHOULDER][0]
        Left_shoulder_x_y_t1 = pose.keypoints[KeypointType.LEFT_SHOULDER][0]

        #vitess left wrist
        vitess_left = vitess(Left_wrist_x_y_t0, Left_wrist_x_y_t1)
        #acceleration left wrist
        acce_left = vitess_left / avg_inference_time

        #vitess Right wrist
        vitess_right = vitess(Right_wrist_x_y_t0, Right_wrist_x_y_t1)
        # acceleration
        acce_right = vitess_right / avg_inference_time

        #angle calculation
        degree_right = getAngle(Right_shoulder_x_y, Right_elbow_x_y, Right_wrist_x_y)
        degree_left = getAngle(Left_shoulder_x_y, Left_elbow_x_y, Left_wrist_x_y)

        # detection d'une Droite 
        if (acce_right > 0.30  and acce_right > acce_left and abs(R_right[0, 1]) > 0.85 and Right_wrist_x_y_t0[1] > nose_x_y[1] and iter_ != db_coups + 1 and degree_right > 170):

            if (R_right[0, 1] > 0):
                db_coups = iter_
                counter_arr[0] = counter_arr[0] + 1
                print("Droite Haut, taux de réussite :",abs(R_right[0, 1])*100,"%")
                flag_incr = 1
                disp_counter = 0

            else:
                db_coups = iter_
                counter_arr[1] = counter_arr[1] + 1
                print("Droite Bas, taux de réussite :",abs(R_right[0, 1])*100,"%")
                flag_incr = 2
                disp_counter = 0

            time.sleep((avg_inference_time / 1000)*40)


        #Detection d'une gauche
        if(acce_left > 0.30 and  acce_left > acce_right and abs(R_left[0, 1])>0.85 and Left_wrist_x_y_t0[1] > nose_x_y[1] and iter_ != db_coups + 1 and degree_left > 170):
            if (R_left[0, 1] < 0):
                db_coups = iter_
                counter_arr[2] = counter_arr[2] + 1
                print("Gauche Haut , taux de réussite :",abs(R_left[0, 1])*100,"%")
                flag_incr = 3
                disp_counter = 0
            else:
                db_coups = iter_
                counter_arr[3] = counter_arr[3] + 1
                print("Gauche Bas, taux de réussite :",abs(R_left[0, 1])*100,"%")
                flag_incr = 4
                disp_counter = 0
            time.sleep((avg_inference_time / 1000)*40)


        #detection d'une uppercut droite
        if (degree_right< 95 and degree_right >15 and  acce_right < 0.50 and  acce_right > 0.1 and acce_right > acce_left and Right_wrist_x_y_t0[1] < nose_x_y[1] and iter_ != db_coups + 1):
            db_coups = iter_
            counter_arr[4] = counter_arr[4] + 1
            print("Uppercut Droite")
            flag_incr = 5
            disp_counter = 0
            time.sleep((avg_inference_time / 1000)*50)


        #detection d'une uppercut gauche
        if (degree_left< 95 and degree_left >15 and  acce_left < 0.50  and acce_left > 0.1and acce_left > acce_right and Left_wrist_x_y_t0[1] < nose_x_y[1] and iter_ != db_coups + 1):
            db_coups = iter_
            counter_arr[5] = counter_arr[5] + 1
            print("Uppercut Gauche")

            flag_incr = 6
            disp_counter = 0
            time.sleep((avg_inference_time / 1000)*50)

        #mise à jour des variables
        Right_wrist_x_y_t0 = Right_wrist_x_y_t1
        Left_wrist_x_y_t0 = Left_wrist_x_y_t1

        Right_shoulder_x_y_t0 = Right_shoulder_x_y_t1
        Left_shoulder_x_y_t0 = Left_shoulder_x_y_t1

        Right_elbow_x_y_t0 = Right_elbow_x_y_t1

    iter_ = iter_ + 1

    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold: continue

        #Offset and scale to source coordinate space.
        kp_x = int((keypoint.point[0] - box_x) * scale_x)
        kp_y = int((keypoint.point[1] - box_y) * scale_y)

        xys[label] = (kp_x, kp_y)

        dwg.add(dwg.circle(center=(int(kp_x), int(kp_y)), r=5,
                           fill='red', fill_opacity=keypoint.score, stroke=color))

def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)


def run(inf_callback, render_callback):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--res', help='Resolution', default='640x480',
                        choices=['480x360', '640x480', '1280x720'])
    parser.add_argument('--videosrc', help='Which video source to use', default='/dev/video0')
    parser.add_argument('--h264', help='Use video/x-h264 input', action='store_true')
    parser.add_argument('--jpeg', help='Use image/jpeg input', action='store_true')
    args = parser.parse_args()

    default_model = 'models/mobilenet/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite'
    if args.res == '480x360':
        src_size = (640, 480)
        appsink_size = (480, 360)
        model = args.model or default_model % (353, 481)
    elif args.res == '640x480':
        src_size = (640, 480)
        appsink_size = (640, 480)
        model = args.model or default_model % (481, 641)
    elif args.res == '1280x720':
        src_size = (1280, 720)
        appsink_size = (1280, 720)
        model = args.model or default_model % (721, 1281)

    #print('Loading model: ', model)
    engine = PoseEngine(model)
    input_shape = engine.get_input_tensor_shape()
    #print("input_shape :",input_shape)
    inference_size = (input_shape[2], input_shape[1])
    #print("inference_size : ",inference_size)
    gstreamer.run_pipeline(partial(inf_callback, engine), partial(render_callback, engine),
                           src_size, inference_size,
                           mirror=args.mirror,
                           videosrc=args.videosrc,
                           h264=args.h264,
                           jpeg=args.jpeg
                           )


def main():
    n = 0
    sum_process_time = 0
    sum_inference_time = 0
    ctr = 0
    fps_counter = avg_fps_counter(30)

    def run_inference(engine, input_tensor):
        return engine.run_inference(input_tensor)

    avg_inference_time = 0
    def render_overlay(engine, output, src_size, inference_box):
        nonlocal n, sum_process_time, sum_inference_time, fps_counter
        global avg_inference_time,svg_canvas,affichage_t0,disp_counter,instruction_executee

        svg_canvas = svgwrite.Drawing('', size=src_size)
        start_time = time.monotonic()
        outputs, inference_time = engine.ParseOutput()
        end_time = time.monotonic()
        n += 1
        sum_process_time += 1000 * (end_time - start_time)
        sum_inference_time += inference_time * 1000

        avg_inference_time = sum_inference_time / n
        text_line = 'PoseNet: %.1fms (%.2f fps) TrueFPS: %.2f' % (
            avg_inference_time, 1000 / avg_inference_time, next(fps_counter)
        )

        shadow_text(svg_canvas, 10, 20, text_line,'16','white')


        if(int(delta_time_setup) <= 10):
            text_setup_time = '%d' % (10 - delta_time_setup)
            shadow_text(svg_canvas, src_size[0]/2, src_size[1]/2, text_setup_time,90,'red')
        elif(int((exercice_time - delta_time_setup + 10)) > 0):
            exercice_time_text = 'temps restant : %d' % (exercice_time - delta_time_setup + 10)
            shadow_text(svg_canvas, 10, 60, exercice_time_text,'16','white')
        else:
            fin_exercice = 'Fin'
            shadow_text(svg_canvas, src_size[0]/2, src_size[1]/2, fin_exercice,90,'red')

            if not instruction_executee:
                # renvoyer les stats au terminal
                print("Droite Haut : ", counter_arr[0])
                print("Droite Bas : ", counter_arr[1])
                print("Gauche Haut : ", counter_arr[2])
                print("Gauche Bas : ", counter_arr[3])
                print("Uppercut Droite : ", counter_arr[4])
                print("Uppercut Gauche : ", counter_arr[5])


            # Mettre à jour la variable pour indiquer que l'instruction a été exécutée
            instruction_executee = True



        #incrustation des coups
        text_incru = switch(flag_incr)
        shadow_text(svg_canvas, 10, 120, text_incru,25,'cyan',flag_incr,disp_counter)
        
        disp_counter = disp_counter + 1

        for pose in outputs:
            draw_pose(svg_canvas, pose, src_size, inference_box)
        return (svg_canvas.tostring(), False)


    run(run_inference, render_overlay)


if __name__ == '__main__':
    main()
