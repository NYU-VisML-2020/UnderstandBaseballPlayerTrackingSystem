import cv2
import json


def extract_frames_with_certain_activities(videoname,input_directory,output_directory,start,end,label):
    vidcap = cv2.VideoCapture(input_directory+videoname+'.mp4')
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv2.imwrite(output_directory+label+'/'+videoname+ "{:04d}".format(count)+".jpg", image)     # save frame as JPG file
        return hasFrames
    sec = start
    frameRate = 0.1 # 0.5: it will capture image in each 0.5 second
    count=0
    success = getFrame(sec)
    while success and sec<end:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)
    # print(videoname,label,start,end)


# videoname='377H4PC2NSEP'
# videoname='QSGJHSYQI11P'
input_directory='/media/felicia/Data/mlb-youtube/continuous_videos/'
output_directory='/media/felicia/Data/mlb-youtube/frames_continuous/'
# extract_frames_with_certain_activities(videoname,input_directory,output_directory,0,200,'unclassified')


count=100
with open('data/mlb-youtube-continuous.json', 'r') as f:
    data = json.load(f)
    for video in list(data.keys())[100:]:
        # data[video]['clip_name']=video
        if count>500:
            break
        duration=data[video]['end']-data[video]['start']
        cap = cv2.VideoCapture(input_directory+video+'.mp4')
        fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
        print(video,count,duration,fps)
        if fps==0:
            continue
        delta=duration-int(cap.get(cv2.CAP_PROP_FRAME_COUNT))/fps
        for activity in data[video]['annotations']:
            start, end=activity['segment']
            label=activity['label']
            extract_frames_with_certain_activities(video,input_directory,output_directory,start-delta,end-delta,label)
        count+=1

        
###############################################

video='O35GBDO4IA6O'
        
input_directory='/media/felicia/Data/mlb-youtube/continuous_videos/'
output_directory = '/media/felicia/Data/mlb-youtube/swing_videos/'


with open('/home/felicia/research/baseballplayer/mlb-youtube/data/mlb-youtube-continuous.json', 'r') as f:
    data = json.load(f)
duration=data[video]['end']-data[video]['start']
cap = cv2.VideoCapture(input_directory+video+'.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)   
delta=duration-int(cap.get(cv2.CAP_PROP_FRAME_COUNT))/fps
for activity in data[video]['annotations']:
    label=activity['label']
    if label=='swing':
        start, end=activity['segment']
        label=activity['label']
        break
print(start,end,duration,delta)

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

ffmpeg_extract_subclip("O35GBDO4IA6O_full.mp4", start-delta, end-delta, targetname="O35GBDO4IA6O_clip.mp4")