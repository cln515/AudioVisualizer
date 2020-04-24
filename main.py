import numpy as np
from scipy import signal
from scipy import interpolate
import cv2

import wave
import math
import sys

#cvt angle to color
def val2color(radangle):
    M_PI = math.pi
    pi_sixtydig = M_PI / 3
    angle = ((radangle / (M_PI*2))- (int)(radangle / (M_PI * 2)))*(M_PI * 2)
    rgb = [0,0,0]
    if (angle >= 0 and angle < pi_sixtydig) :
        val = (angle - pi_sixtydig*0)/ pi_sixtydig
        rgb[0] = 255
        rgb[1] = 255*val
        rgb[2] = 0
    elif(angle >= pi_sixtydig*1 and angle < pi_sixtydig*2) :
        val = (angle - pi_sixtydig * 1) / pi_sixtydig
        rgb[0] = 255 *(1 - val)
        rgb[1] = 255 
        rgb[2] = 0
	
    elif (angle >= pi_sixtydig * 2 and angle < pi_sixtydig * 3): 
        val = (angle - pi_sixtydig * 2) / pi_sixtydig
        rgb[0] = 0 
        rgb[1] = 255
        rgb[2] = 255 * ( val)
	
    elif (angle >= pi_sixtydig * 3 and angle < pi_sixtydig * 4) :
        val = (angle - pi_sixtydig * 3) / pi_sixtydig
        rgb[0] = 0 
        rgb[1] = 255 * (1 - val)
        rgb[2] = 255
	
    elif (angle >= pi_sixtydig * 4 and angle < pi_sixtydig * 5) :
        val = (angle - pi_sixtydig * 4) / pi_sixtydig
        rgb[0] = 255 * ( val)
        rgb[1] = 0
        rgb[2] = 255 
	
    elif (angle >= pi_sixtydig * 5 and angle < pi_sixtydig * 6) :
        val = (angle - pi_sixtydig * 5) / pi_sixtydig
        rgb[0] = 255 
        rgb[1] = 0
        rgb[2] = 255 * (1 - val)
	
    return (rgb[0],rgb[1],rgb[2])

global circle_num
global idx_range_sum
global temporal_cnt
global col_change_speed
col_change_speed = 0.01
circle_num = 5



def drawCircles(points_array,color, alpha=1.0):
    global circle_num
    global temporal_cnt
    for i in range(int(len(points_array))):
        for j in range(int(len(points_array[i]))):
            if j==int(len(points_array[i]))-1:
                cv2.line(base,points_array[i][j],points_array[i][0],np.multiply(alpha,val2color(color[i]+temporal_cnt*col_change_speed)),2)   
            else:
                cv2.line(base,points_array[i][j],points_array[i][j+1],np.multiply(alpha,val2color(color[i]+temporal_cnt*col_change_speed)),2)

def cnt_where(idx,idx_range):
    cnt = 1
    val = idx_range_sum[cnt]
    while idx >= val:
        cnt += 1
        val =idx_range_sum[cnt]
    return cnt-1

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = 8*(x-min)/(max-min)
    return result

def base_update(mean,cur):
    add_v = ((mean+8)/2 - cur) * 0.05
    cur = cur + add_v
    if cur > 8:
        cur = 8
    return cur

def peak_update(max,cur):
    add_v = ((max)/2 - cur) * 0.05
    cur = cur + add_v
    if cur > 2:
        cur = 2
    if cur < 0:
        cur = 0
    return cur

#spline fitting
def spline3(pointarray,point_num,deg):
    x = []
    y = []
    for i in range(len(pointarray)):
        x.append(pointarray[i][0])
        y.append(pointarray[i][1])
    tck,u = interpolate.splprep([x,y],k=deg,s=0) 
    u = np.linspace(0,1,num=point_num,endpoint=True) 
    spline = interpolate.splev(u,tck)
    spline_points = []
    for i in range(len(spline[0])):
        spline_points.append((int(spline[0][i]),int(spline[1][i])))
    return spline_points

# number of afterimage
num = 10

# input wav file and output mp4 file
wavf = sys.argv[1]
outmp4 = sys.argv[2]

#output length (second)
length_video = int(sys.argv[3])

#read wav data
wr = wave.open(wavf, 'r')
ch = wr.getnchannels()
width = wr.getsampwidth()
fr = wr.getframerate()
fn = wr.getnframes()

data = wr.readframes(wr.getnframes())
wr.close()
X = np.frombuffer(data, dtype=np.int16)

if ch == 2:
    l_channel = (1.0/255)* X[::ch]
    r_channel = (1.0/255)* X[1::ch]

print(l_channel)
print(len(r_channel))
print("Channel: ", ch)
print("Sample width: ", width)
print("Frame Rate: ", fr)
print("Frame num: ", fn)
print("Params: ", wr.getparams())
print("Total time: ", 1.0 * fn / fr)

# set output mp4
frame_rate = 30.0
fmt = cv2.VideoWriter_fourcc('m','p','4','v')
writer = cv2.VideoWriter(sys.argv[2],fmt,frame_rate, (640,480))

length_video = int(length_video * frame_rate)

# fft parameter (window size)
window_size = 8092

# array for drawing points
points_v = []

# visualization parameters (maximum and minimum db, border index of frequency)
base_peak = [2,2,2,2,2]
base_val=[8,8,8,8,8]
idx_range_sum = [2,40,80,200,400,520]
temporal_cnt = 0
for j in range(length_video):
    print("\r"+str(int(j/30))+ "s rendered", end="")
    init_pos =int( j * 44100/ 30)

    # spectrum
    freq , pw = signal.welch(l_channel[init_pos:init_pos+window_size],fr, nperseg=4096)
    
    # update max and min of db
    pw_db = -np.log10(pw)
    for i in range(circle_num):
        base_peak[i]=peak_update(np.min(pw_db[idx_range_sum[i]+1:idx_range_sum[i+1]]),base_peak[i])
        base_val[i]=base_update(np.mean(pw_db[idx_range_sum[i]+1:idx_range_sum[i+1]]),base_val[i])

    base = np.zeros((480,640,3), np.uint8)
    v = len(points_v)
    prange = min(len(points_v),num)

    # draw afterimage
    for k in range(prange):
        drawCircles(points_v[v-(prange-k)-1], base_val,(k)/num )

    points_a = []
    points = []
    x_s = []
    y_s = []
    prev = -1

    # spectrum 2 drawing points ((frec,power)-->(r,theta))
    for i in range(0,idx_range_sum[circle_num]+1):
        if i <= idx_range_sum[0]:
            continue
        idx = cnt_where(i-1,idx_range_sum)

        if not prev == -1 and not prev == idx:
            points_a.append(spline3(points,5*len(points),3))
            points = []
        prev = idx
        theta = -(freq[i]- freq[idx_range_sum[idx]]) / (freq[idx_range_sum[idx + 1]] - freq[idx_range_sum[idx]]) * math.pi  *2   
        v1 = pw_db[i] 
        if v1 < base_peak[idx] :
            v1 = base_peak[idx]
        if v1 > base_val[idx] :
            v1 = base_val[idx]
        r1 = 40 + 40* idx + 30 * (base_val[idx]- v1) / (base_val[idx] - base_peak[idx])
        p1=(320 + int(r1 * math.cos(theta)),240 + int(r1 * math.sin(theta)))

        x_s.append(320 + int(r1 * math.cos(theta)))
        y_s.append(240 + int(r1 * math.sin(theta)))
        
        points.append(p1)

    # spline fitting
    points_a.append(spline3(points,5*len(points),3))        
    # draw current frame
    drawCircles(points_a, base_val)
    points_v.append(points_a)
    # render frame
    writer.write(base)
    temporal_cnt +=1
print("\nfinished!")
