from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math
O=Path(__file__).resolve().parent
im=Image.new('RGB',(2400,1640),'white');d=ImageDraw.Draw(im)
F='/System/Library/Fonts/AppleSDGothicNeo.ttc'
ink='#173049';muted='#63768A';blue='#246FD0';light='#EAF3FD';gray='#A6B5C6';line='#DCE7F2';green='#178477'
def font(n):return ImageFont.truetype(F,n)
def t(x,y,s,n=32,c=ink):d.text((x,y),s,font=font(n),fill=c)
def ct(x,y,w,s,n=32,c=ink):t(x+(w-d.textlength(s,font=font(n)))/2,y,s,n,c)
def box(x,y,w,h,c=light,r=20,outline=None):d.rounded_rectangle((x,y,x+w,y+h),radius=r,fill=c,outline=outline,width=3)
def arr(x1,y1,x2,y2,c=blue):
 d.line((x1,y1,x2,y2),fill=c,width=5);a=math.atan2(y2-y1,x2-x1)
 d.polygon([(x2,y2),(x2-15*math.cos(a-.5),y2-15*math.sin(a-.5)),(x2-15*math.cos(a+.5),y2-15*math.sin(a+.5))],fill=c)
def car(x,y,s=1,c=blue):
 def P(a,b):return(x+a*s,y+b*s)
 d.polygon([P(15,40),P(35,8),P(100,8),P(125,40)],fill=c)
 d.rounded_rectangle((*P(0,35),*P(140,83)),radius=int(10*s),fill=c)
 d.polygon([P(39,17),P(61,17),P(61,39),P(27,39)],fill='white')
 d.polygon([P(70,17),P(95,17),P(112,39),P(70,39)],fill='white')
 for a in [20,100]:d.ellipse((*P(a,66),*P(a+26,92)),fill=ink)
def frames(x,y):
 for dx,dy in [(26,-22),(13,-11),(0,0)]:
  box(x+dx,y+dy,245,158,'#F6F9FC',12,line)
  d.line((x+dx+18,y+dy+133,x+dx+226,y+dy+133),fill=line,width=3)
 car(x+50,y+43,.95)
def roi(x,y):
 frames(x,y);d.rectangle((x+38,y+32,x+202,y+140),outline=green,width=5)
def network(x,y):
 for a in range(3):
  for b in range(3):
   for z in range(3):
    if a<2:d.line((x+a*52,y+b*46,x+(a+1)*52,y+z*46),fill='#B7D2EF',width=2)
 for a in range(3):
  for b in range(3):d.ellipse((x+a*52-9,y+b*46-9,x+a*52+9,y+b*46+9),fill=blue)
def pill(x,y,w,s):box(x,y,w,65,blue,32);ct(x,y+11,w,s,36,'white')

t(80,48,'SlowFast 기반 차량 충돌 탐지',68)
t(84,139,'영상 속 움직임을 학습하고, 사고 의심 구간을 찾아냅니다',33,muted)
pill(80,225,1030,'모델 학습 과정');pill(1200,225,1120,'성능 비교')
# Training schematic
frames(135,365);ct(100,549,340,'충돌·정상 영상',38)
arr(450,448,588,448)
roi(651,365);ct(611,549,340,'차량 영역 추출',38)
arr(774,607,774,660)
# dual path graphic, no unverified training hyperparameters
box(114,693,400,122,light);ct(114,712,400,'Slow',42,blue);ct(114,766,400,'형태·공간 특징',30)
box(114,850,400,122,'#D9EAFD');ct(114,869,400,'Fast',42,blue);ct(114,923,400,'움직임·시간 변화',30)
d.line((774,660,774,674,73,674,73,910),fill=blue,width=4)
arr(73,754,105,754);arr(73,910,105,910)
arr(315,843,315,822)
arr(524,754,581,754);arr(524,910,581,910)
d.line((581,754,581,910),fill=blue,width=4);arr(581,832,636,832)
network(680,785)
ct(633,908,250,'특징 결합·학습',29)
arr(813,832,880,832)
box(896,748,197,170,blue);ct(896,785,197,'SlowFast',34,'white');ct(896,837,197,'R50',37,'white')
ct(96,1029,1010,'영상 → 차량 영역 → 두 경로 학습 → 충돌 분류',31,muted)
# Two metric panels with exact, user-provided values. Honest zero baselines.
models=['S3D','X3D-S','SlowFast-R50']
def chart(top,title,values,limit,labels):
 t(1220,top,title,37)
 left=1312;right=2280;base=top+274;plot_top=top+70;ph=204
 for frac in [0,.5,1]:
  yy=base-ph*frac;d.line((left,yy,right,yy),fill=line,width=2)
  lab=str(int(limit*frac)) if limit==100 else ('0' if frac==0 else str(frac))
  t(1230,yy-17,lab,27,muted)
 for i,(m,v,l) in enumerate(zip(models,values,labels)):
  x=1400+i*325;bw=145;bh=ph*v/limit;c=blue if i==2 else gray
  d.rectangle((x,base-bh,x+bw,base),fill=c)
  ct(x-36,base-bh-43,bw+72,l,34,ink)
  ct(x-72,base+21,bw+144,m,29,blue if i==2 else ink)
chart(330,'검증 정확도 (%)',[90.9,81.8,100],100,['90.9%','81.8%','100%'])
chart(755,'사고 F1 (0–1)',[.75,.5,1],1,['0.75','0.50','1.00'])
# service outcome flow
d.line((80,1170,2320,1170),fill=line,width=3)
t(84,1210,'웹에서는 이렇게 사용합니다',41)
steps=[(95,'영상 등록'),(668,'차량 선택'),(1241,'AI 분석'),(1814,'의심 구간·CAM')]
for idx,(x,txt) in enumerate(steps):
 box(x,1295,470,173,'#F5F9FE',18)
 if idx==0:
  d.rounded_rectangle((x+37,1334,x+136,1419),radius=10,outline=blue,width=5)
  d.polygon([(x+72,1355),(x+72,1399),(x+108,1377)],fill=blue)
 elif idx==1:
  car(x+30,1342,.8);d.rectangle((x+24,1331,x+150,1430),outline=green,width=4)
 elif idx==2:network(x+43,1340)
 else:
  d.rounded_rectangle((x+26,1332,x+148,1424),radius=9,outline=blue,width=4)
  d.ellipse((x+57,1353,x+115,1407),fill='#F8BA62');d.ellipse((x+76,1370,x+103,1399),fill='#DF604B')
 t(x+176,1357,txt,36)
 if idx<3:arr(x+485,1380,x+553,1380)
t(84,1520,'선택 모델  SlowFast-R50',34,blue)
t(675,1520,'검증 정확도 100%     사고 F1 1.00',37)
t(84,1584,'성능 수치: 사용자 제공 검증 결과 · 학습 흐름은 개념도 · 검증 표본 수·평가 조건 미제공',25,muted)
im.save(O/'slowfast_visual_summary.png',dpi=(200,200))
print(O/'slowfast_visual_summary.png')
