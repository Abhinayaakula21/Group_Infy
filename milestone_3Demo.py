# ===================== IMPORT LIBRARIES =====================

import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import platform
from ctypes import POINTER, cast

# ===================== GRAPH SETTINGS =====================

MIN_DISTANCE = 30
MAX_DISTANCE = 250
GRAPH_HISTORY = 90

history = deque(maxlen=GRAPH_HISTORY)

# ===================== MEDIAPIPE SETUP =====================

# ===== ALREADY EXPLAINED IN MILESTONE 2 =====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===================== SYSTEM VOLUME =====================

# ===== NEW IN MILESTONE 3 =====
def get_volume_controller():

    if platform.system() != "Windows":
        raise RuntimeError("Volume control only supported on Windows")

    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

    return cast(interface, POINTER(IAudioEndpointVolume))

volume_controller = get_volume_controller()

# ===================== DISTANCE → VOLUME =====================

def map_distance_to_volume(distance):

    distance = np.clip(distance, MIN_DISTANCE, MAX_DISTANCE)

    return np.interp(distance, [MIN_DISTANCE, MAX_DISTANCE], [0, 100])

# ===================== GRAPH FUNCTION =====================

def build_graph(distance, volume):

    graph = np.full((420,640,3),245,np.uint8)

    x0,y0 = 90,360
    x1,y1 = 580,70

    # Graph background
    cv2.rectangle(graph,(x0,y1),(x1,y0),(220,220,220),-1)
    cv2.rectangle(graph,(x0,y1),(x1,y0),(60,60,60),2)

    # ===== GRID LINES =====

    for i in range(0,101,10):
        y = int(np.interp(i,[0,100],[y0,y1]))
        cv2.line(graph,(x0,y),(x1,y),(200,200,200),1)
        cv2.putText(graph,str(i),(50,y+5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(80,80,80),1)

    for d in range(int(MIN_DISTANCE),int(MAX_DISTANCE)+1,20):
        x = int(np.interp(d,[MIN_DISTANCE,MAX_DISTANCE],[x0,x1]))
        cv2.line(graph,(x,y0),(x,y1),(200,200,200),1)
        cv2.putText(graph,str(d),(x-12,y0+20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(80,80,80),1)

    # Mapping line
    cv2.line(graph,(x0,y0),(x1,y1),(0,170,120),3)

    # ===== HISTORY LINE =====

    history.append((distance,volume))

    points=[]

    for d,v in history:

        px=int(np.interp(d,[MIN_DISTANCE,MAX_DISTANCE],[x0,x1]))
        py=int(np.interp(v,[0,100],[y0,y1]))

        points.append((px,py))

    for i in range(1,len(points)):
        cv2.line(graph,points[i-1],points[i],(255,120,0),2)

    # ===== CURRENT POINT =====

    cx=int(np.interp(distance,[MIN_DISTANCE,MAX_DISTANCE],[x0,x1]))
    cy=int(np.interp(volume,[0,100],[y0,y1]))

    cv2.circle(graph,(cx,cy),7,(0,0,255),-1)
    cv2.circle(graph,(cx,cy),10,(255,255,255),2)

    # ===== TEXT LABELS =====

    cv2.putText(graph,"Distance vs Volume Mapping",
                (200,35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(40,40,40),2)

    cv2.putText(graph,"Volume %",
                (15,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,(50,50,50),1)

    cv2.putText(graph,"Distance (pixels)",
                (250,415),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,(50,50,50),1)

    return graph


# ===================== CAMERA =====================

cap = cv2.VideoCapture(0)

# ===================== MAIN LOOP =====================

while True:

    ret,frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame,1)

    # ===== ALREADY EXPLAINED IN MILESTONE 2 =====
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    distance = MIN_DISTANCE

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

        h,w,_ = frame.shape

        # ===== ALREADY EXPLAINED IN MILESTONE 2 =====
        x1 = int(hand_landmarks.landmark[4].x * w)
        y1 = int(hand_landmarks.landmark[4].y * h)

        x2 = int(hand_landmarks.landmark[8].x * w)
        y2 = int(hand_landmarks.landmark[8].y * h)

        # ===== ALREADY EXPLAINED IN MILESTONE 2 =====
        distance = math.hypot(x2-x1,y2-y1)

        cv2.circle(frame,(x1,y1),8,(0,255,0),-1)
        cv2.circle(frame,(x2,y2),8,(0,255,0),-1)

        cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)

    # ===== NEW IN MILESTONE 3 =====
    volume = map_distance_to_volume(distance)

    volume_controller.SetMasterVolumeLevelScalar(volume/100,None)

    # ===================== VOLUME BAR =====================

    bar = int(np.interp(volume,[0,100],[350,150]))

    cv2.rectangle(frame,(40,150),(80,350),(0,255,0),2)
    cv2.rectangle(frame,(40,bar),(80,350),(0,255,0),-1)

    cv2.putText(frame,f"{int(volume)} %",
                (30,120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),3)

    cv2.putText(frame,f"Distance: {int(distance)} px",
                (150,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,(0,255,255),2)

    # ===================== GRAPH =====================

    graph = build_graph(distance,volume)

    cv2.imshow("Milestone 3 - Volume Control",frame)
    cv2.imshow("Distance vs Volume Graph",graph)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===================== CLEANUP =====================

cap.release()
cv2.destroyAllWindows()
hands.close()