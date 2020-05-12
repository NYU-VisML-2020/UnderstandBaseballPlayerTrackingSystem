import os 
import json
import pickle
import numpy as np 

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from itertools import repeat

import pandas as pd

import seaborn as sns
import scipy.stats as stats

from sklearn.metrics import pairwise_distances_chunked

JSON_DIR='/media/felicia/Data/baseballplayers/jsons/'
PICKLE_DIR='/media/felicia/Data/baseballplayers/pickles/'
PATH_TO_TEST_IMAGES_DIR = '/media/felicia/Data/mlb-youtube/frames_continuous/swing'

# videonames=pickle.load(open(PICKLE_DIR+'swing_videos.pkl','rb'))

# nvideos=10
nframes=19
max_boxes=10

HEIGHT=720
WIDTH=1280

# videos_labeled=videonames[:9]+[videonames[10]]
# videos_labeled=['O35GBDO4IA6O', 'YSVQ8K3F6GOP', 'HNSGD55ICW30', 'DJZAPL1KKGBX', '3II3NWFATLG6', 'KWS1I2C5Q0Y4', 'O3ZIY1X48AEV', 'BNPP45GVM0QJ', '0FXJX72YFEZK', 'KNY70TW3FMX4', 
#  'MN0RZJGOBTJZ', 'Q9C23P705TRU', 'GFMYSCF7BYK8', 'I3DEK2MIEZ6T', 'T2IYANC1STR9', 'BGVNASU9KTK8', 'YOCVMAVF7YHI', '48FTCTOBOV4K', 'KQMRHE3CL32Y', 'D1SJ96G3YWVM', 
#  'N7VKVX43NM33', '3HKN4TBFV03M', 'OW1WHO74RNX2', 'EHJSEDXCCG78', 'MCE1ZFDBM4PR', 'GHD9ZYPBWH9G', 'LIOI1CMB6QXL', 'B0VMWK3L6O6S', 'D4HUIBMV4Y11', 'MG471B8BIBZ5',
#  'W0Q357CX1ZUI', 'HXIYL93XDATX', '28AUMAQ2FO5S', 'VHGQQZJQDR8H', 'IV1RGJLV1ZSS', 'WZ5OGK2KKBXV', '41EIMJH58QA9', 'C6WOI82993OF', '04WR4VWZ0SIX', 'IKCRBUUI2FOK',
#  'PEMSZNYHTZ27', 'BWISZUV6XC2B', 'OQGZJBOCMYTQ', '9T9WA3OTCKMZ', 'KDV4OD86Z155', 'ATV6SD1T4APF', 'SK0RVHTBCGJP', '2I38PMQAZKOE', 'GFL3GOUXKB7C', 'VDOMD6VJ0AP4', 
#  'D8VI1WQ5GFI0', '2YNUMO3SW49Q', 'Y5UXMCXUPBGY', 'J3ERR9DPIRBV', 'JUHQMF1NKBUE', 'ULP7ZMSXZQWG', 'S0M5M2LFVJZE', '20FOCYBEKWII', '460B5936QNYX', '3VMF7ZMRB0Q6',
#  'L4P6DIXQ29IU', '5ZPWAUXTSGE3', 'J9VE9EPIKDNN', '41FJ1IVVWLR7', 'WFZWNZ6M6RPC', '9TBNL5Z0IEUC', '6Z5T8NHCRDN7', 'RWL516VQXP3M', 'PC7AQD2WAULV', '6HRWA3MVHM4M', 
#  '8G08KQLD9ZV4', 'QNURL5QJZHKH', '08X6PV2MZTA6', 'BHGW6GBF4NCL', 'AZO63JPJXTVD', '0XWRT1PVIS54', '903PQ76PTHTI', 'Y61YSRZJJFO3', '521EJBXVDS6M', 'QD8WF4HTJ6UU', 
#  'OJVGHFFIYPZA', 'KNJDDL9HO65J', '018TV9B81MWH', 'T8ZKDHHXV649']

videos_labeled=['O35GBDO4IA6O', 'YSVQ8K3F6GOP', 'HNSGD55ICW30', 'DJZAPL1KKGBX', '3II3NWFATLG6', 'KWS1I2C5Q0Y4', 'O3ZIY1X48AEV', 'BNPP45GVM0QJ', '0FXJX72YFEZK', 'KNY70TW3FMX4', 
                'MN0RZJGOBTJZ', 'Q9C23P705TRU', 'GFMYSCF7BYK8', 'I3DEK2MIEZ6T', 'T2IYANC1STR9', 'BGVNASU9KTK8', 'YOCVMAVF7YHI', '48FTCTOBOV4K', 'KQMRHE3CL32Y', 'D1SJ96G3YWVM', 
                'N7VKVX43NM33', '3HKN4TBFV03M', 'OW1WHO74RNX2', 'EHJSEDXCCG78', 'MCE1ZFDBM4PR', 'GHD9ZYPBWH9G', 'LIOI1CMB6QXL', 'B0VMWK3L6O6S', 'D4HUIBMV4Y11', 'MG471B8BIBZ5',
                'W0Q357CX1ZUI', 'HXIYL93XDATX', '28AUMAQ2FO5S', 'VHGQQZJQDR8H', 'IV1RGJLV1ZSS', 'WZ5OGK2KKBXV', '41EIMJH58QA9', 'C6WOI82993OF', '04WR4VWZ0SIX', 'IKCRBUUI2FOK',
                'PEMSZNYHTZ27', 'BWISZUV6XC2B', 'OQGZJBOCMYTQ', '9T9WA3OTCKMZ', 'KDV4OD86Z155', 'ATV6SD1T4APF', 'SK0RVHTBCGJP', '2I38PMQAZKOE', 'GFL3GOUXKB7C', 'VDOMD6VJ0AP4', 
                'D8VI1WQ5GFI0', '2YNUMO3SW49Q', 'Y5UXMCXUPBGY', 'J3ERR9DPIRBV', 'JUHQMF1NKBUE', 'ULP7ZMSXZQWG', 'S0M5M2LFVJZE', '20FOCYBEKWII', '460B5936QNYX', '3VMF7ZMRB0Q6',
                'L4P6DIXQ29IU', '5ZPWAUXTSGE3', 'J9VE9EPIKDNN', '41FJ1IVVWLR7', 'WFZWNZ6M6RPC', '9TBNL5Z0IEUC', '6Z5T8NHCRDN7', 'RWL516VQXP3M', 'PC7AQD2WAULV', '6HRWA3MVHM4M', 
                '8G08KQLD9ZV4', 'QNURL5QJZHKH', '08X6PV2MZTA6', 'BHGW6GBF4NCL', 'AZO63JPJXTVD', '0XWRT1PVIS54', '903PQ76PTHTI', 'Y61YSRZJJFO3', '521EJBXVDS6M', 'QD8WF4HTJ6UU', 
                'OJVGHFFIYPZA', 'KNJDDL9HO65J', '018TV9B81MWH', 'T8ZKDHHXV649', '1CR6OV5J0TPW', 'JVDH7IOFMHY6', 'CFJEVB0W7B3U', '5TZVJJXCZNQ5', 'M0TVZYGKNN69', 'M2EYMNRVSN5T', 
                '7JRHFLNHU80M', 'JH5WG9MPP54W', 'CENRP036GY6L', 'T33VSMQ3ZHT0', '6WK6R6J7SYCM', 'JNUQ34NAASOU', 'OJ6KPMBG8NGF', 'S65R0AMA8EUJ', '3WGYMYXFWLSZ', 'XED6N7VRYWA2']


nvideos=len(videos_labeled)
print(nvideos)
# print(videos_labeled)

# Temporal
player_scores=np.zeros((4,nvideos,nframes),dtype=np.float64)
player_areas=np.zeros((4,nvideos,nframes),dtype=np.float64)
player_centers=np.zeros((4,nvideos,nframes,2),dtype=np.float64)

for idx_v,v in enumerate(videos_labeled):
    bbox_json=json.load(open(JSON_DIR+v+'.json','r'))[v]
    scores=bbox_json['accuracy'] # 19*10
    labels=bbox_json['label'] # 19*10
    areas=bbox_json['area']
    bboxs=bbox_json['bbox'] # 19*10*4,ymin, xmin, ymax, xmax = box
    # print(v)
    for i in range(nframes):
        for j in range(max_boxes):
            if labels[i][j]==0:
                continue
            player=labels[i][j] # int
            player_scores[labels[i][j]-1][idx_v][i]=scores[i][j] # float
            player_areas[labels[i][j]-1][idx_v][i]= areas[i][j]  if player>0 else 0
            player_xs=(bboxs[i][j][1]+bboxs[i][j][3])*.5 # 
            player_ys=(bboxs[i][j][0]+bboxs[i][j][2])*.5
            player_centers[labels[i][j]-1][idx_v][i]= np.array([player_xs,player_ys])

# Average for each video


ave_scores=np.zeros((4,nvideos),dtype=np.float64)
ave_centerdist=np.zeros((4,nvideos),dtype=np.float64)
ave_areadist=np.zeros((4,nvideos),dtype=np.float64)

for i in range(4):
    for j in range(nvideos):
        ave_scores[i][j]=np.sum(player_scores[i][j])/np.count_nonzero(player_scores[i][j]) if np.count_nonzero(player_scores[i][j])>0 else 0
        nonzero_centers=player_centers[i][j][player_scores[i][j]>0]
        nonzero_areas=np.expand_dims(player_areas[i][j][player_areas[i][j]>0],axis=1)
        print(i,j, nonzero_areas.shape)
        ave_centerdist[i][j]=next(pairwise_distances_chunked(nonzero_centers)).mean() if nonzero_centers.shape[0]> 0 else 0
        ave_areadist[i][j]=next(pairwise_distances_chunked(nonzero_areas)).mean() if nonzero_areas.shape[0]> 0 else 0

baseballplayer=['Pitcher','Batter','Catcher','Umpire']
colors=plt.get_cmap('Set2')

"""
Auccracy
"""


# """
# scatter plot

# ax.set_color_cycle(colors.colors)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
# fig.xlabel("i-th frame in 'swing' videos")
# fig.ylabel("object detection scores")

fig,ax=plt.subplots(2,2)
x_frames=np.array([x for x in range(nframes)]*nvideos)


for i in range(4):
    player_scores_=player_scores[i].flatten()
    player_areas_=player_areas[i].flatten()
    ax[int(i/2),i%2].scatter(x_frames,player_scores_, s=1+player_areas_*2.5e3,c=colors.colors[i],alpha=0.1,label=baseballplayer[i]) # s=1+player_areas_*2.5e3
    ax[int(i/2),i%2].legend(loc='center left', bbox_to_anchor=(0.8, 1.05))
    ax[int(i/2),i%2].set(xlabel="i-th frame in 'swing' videos",ylabel='object detection scores')
    plt.sca(ax[int(i/2),i%2])
    plt.xticks(np.arange(0,18+1,1))

plt.suptitle("Object Detection on Baseball Players")
plt.show()

# fig.clear()
# """

# violin plot


# ithframe=[str(x) for x in range(nframes)]
ithframe=np.array([x for x in range(nframes)]*nvideos)
# color_label={}
# for i in range(4):
#     color_label[baseballplayer[i]]=colors.colors[i]

sns.set()
fig,axes=plt.subplots(2,2)
for i in range(4):
    player_scores_=player_scores[i].flatten() # 100*19--> [...]
    player_=[baseballplayer[i]]* (nframes*nvideos)
    df_scores=pd.DataFrame(list(zip(player_scores_,ithframe,player_)),columns=['Score','Frame','Player'])
    df_nonzero=df_scores[df_scores.Score>0]
    sns.violinplot(x='Frame',y='Score',data=df_nonzero, inner='quartile', cut=0.5,scale='count', linewidth=1,color=colors.colors[i],ax=axes[int(i/2),i%2],saturation=1) #color=colors.colors[i]
    patch = mpatches.Patch(color=colors.colors[i])
    # fake_handles=repeat(patch,1)
    axes[int(i/2),i%2].legend([patch],[baseballplayer[i]],loc='center left', bbox_to_anchor=(0.8, 1.05))
    plt.sca(axes[int(i/2),i%2])
    plt.yticks(np.arange(0,1.1,0.25))

plt.suptitle("Object Detection Scores")
plt.show()


# box plot
fig,ax=plt.subplots(2,2)
x_frames=np.array([x for x in range(nframes)])
for i in range(4):
    player_scores_=player_scores[i].transpose().tolist() # nframes * nvideos
    player_areas_=player_areas[i].transpose().tolist()
    ax[int(i/2),i%2].boxplot(player_scores_,positions=x_frames,medianprops=dict(color=colors.colors[i]))
    ax[int(i/2),i%2].legend(loc='center left', bbox_to_anchor=(0.8, 1.05))
    ax[int(i/2),i%2].set(xlabel="i-th frame in 'swing' videos",ylabel='object detection scores')
    ax[int(i/2),i%2].set_title(baseballplayer[i])
    plt.sca(ax[int(i/2),i%2])
    plt.xticks(np.arange(0,18+1,1))
plt.suptitle("Object Detection on Baseball Players")
plt.show()


"""
Area
"""
fig,ax=plt.subplots(2,2)
x_frames=np.array([x for x in range(nframes)]*nvideos)


for i in range(4):
    # player_scores_=player_scores[i].flatten()
    player_areas_=player_areas[i].flatten()
    ax[int(i/2),i%2].scatter(x_frames,player_areas_,c=colors.colors[i],alpha=0.3,label=baseballplayer[i]) # s=1+player_areas_*2.5e3
    ax[int(i/2),i%2].legend(loc='center left', bbox_to_anchor=(0.8, 1.05),framealpha=0.5)
    ax[int(i/2),i%2].set(xlabel="i-th frame in 'swing' videos",ylabel='Bounding box area')
    plt.sca(ax[int(i/2),i%2])
    plt.xticks(np.arange(0,18+1,1))
    plt.yticks(np.arange(0.00,0.18,0.025))

plt.suptitle("Object Detection on Baseball Players")
plt.show()


# violin box

sns.set()
fig,axes=plt.subplots(2,2)
for i in range(4):
    player_areas_=player_areas[i].flatten() # 100*19--> [...]
    player_=[baseballplayer[i]]* (nframes*nvideos)
    df_areas=pd.DataFrame(list(zip(player_areas_,ithframe,player_)),columns=['Area','Frame','Player'])
    # df_nonzero=df_areas[df_areas.Area>0]
    sns.violinplot(x='Frame',y='Area',data=df_areas, inner='quartile', cut=0.5,scale='count', linewidth=1,color=colors.colors[i],ax=axes[int(i/2),i%2],saturation=1) #color=colors.colors[i]
    patch = mpatches.Patch(color=colors.colors[i])
    axes[int(i/2),i%2].legend([patch],[baseballplayer[i]],loc='center left', bbox_to_anchor=(0.8, 1.05))
    plt.sca(axes[int(i/2),i%2])
    plt.yticks(np.arange(0.00,0.18,0.025))

plt.suptitle("Normalized Bounding Box Areas")
plt.show()

"""
Position(center)
"""
sns.set(style="white", color_codes=True)
for i in range(4):
    ave_scores_=ave_scores[i] # 100
    ave_centerdist_=ave_centerdist[i] # 100
    df_joint=pd.DataFrame(list(zip(ave_scores_,ave_centerdist_)),columns=['score','ave_dist'])
    df_nonzero=df_joint[df_joint.score>0]
    print(i,df_nonzero.shape)
    jp=sns.jointplot(x='score',y='ave_dist',data=df_nonzero, kind='hex',color=colors.colors[i],height=8) #color=colors.colors[i]
    jp.annotate(stats.pearsonr,loc='upper left',fontsize=9)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    cbar_ax = jp.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
    plt.colorbar(cax=cbar_ax)
    jp.fig.suptitle('Object Detection On '+baseballplayer[i],y=0.85)
    plt.sca(jp.fig.axes[0])
    plt.xticks(np.arange(0.3,1.1,0.1))
    plt.yticks(np.arange(0.00,0.145,0.01))

plt.show()




"""
1 (100, 2)
1 (100, 2)
2 (100, 2)
3 (93, 2)


"""


"""
area with score: joint distribution+helix
"""

# with sns.axes_style('white'):
#     sns.jointplot("x", "y", data, kind='kde')


sns.set(style="white", color_codes=True)
# sns.axes_style()
# fig,axes=plt.subplots(2,2)
for i in range(4):
    player_scores_=player_scores[i].flatten() # 100*19--> [...]
    player_areas_=player_areas[i].flatten() 
    df_joint=pd.DataFrame(list(zip(player_scores_,player_areas_)),columns=['score','bbox_area'])
    df_nonzero=df_joint[df_joint.score>0]
    print(i,df_nonzero.shape)
    jp=sns.jointplot(x='score',y='bbox_area',data=df_nonzero, kind='hex',color=colors.colors[i],height=8) #color=colors.colors[i]
    jp.annotate(stats.pearsonr,loc='upper left',fontsize=9)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    cbar_ax = jp.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
    plt.colorbar(cax=cbar_ax)
    jp.fig.suptitle('Object Detection On '+baseballplayer[i],y=0.85)
    plt.sca(jp.fig.axes[0])
    plt.xticks(np.arange(0.3,1.1,0.1))
    plt.yticks(np.arange(0.00,0.15,0.01))

plt.show()



sns.set(style="white", color_codes=True)
for i in range(4):
    ave_scores_=ave_scores[i] # 100
    ave_areadist_=ave_areadist[i] # 100
    df_joint=pd.DataFrame(list(zip(ave_scores_,ave_areadist_)),columns=['score','ave_dist'])
    df_nonzero=df_joint[df_joint.score>0]
    print(i,df_nonzero.shape)
    jp=sns.jointplot(x='score',y='ave_dist',data=df_nonzero, kind='hex',color=colors.colors[i],height=8) #color=colors.colors[i]
    jp.annotate(stats.pearsonr,loc='upper left',fontsize=9)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    cbar_ax = jp.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
    plt.colorbar(cax=cbar_ax)
    jp.fig.suptitle('Object Detection On '+baseballplayer[i],y=0.85)
    # plt.sca(jp.fig.axes[0])
    # plt.xticks(np.arange(0.3,1.1,0.1))
    # plt.yticks(np.arange(0.00,0.145,0.01))

plt.show()


"""
0 (1790, 2)
1 (1842, 2)
2 (1793, 2)
3 (739, 2)
"""