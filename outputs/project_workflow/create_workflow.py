from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT=Path(__file__).resolve().parent
W,H=2400,1900
im=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(im)
FONT='/System/Library/Fonts/AppleSDGothicNeo.ttc'
C={'ink':'#14283D','muted':'#52677C','blue':'#226DC4','pale':'#F3F8FD','line':'#D8E4F0','green':'#127A65','gp':'#ECF7F2','orange':'#A56116','op':'#FFF7E9'}
def f(n): return ImageFont.truetype(FONT,n)
def text(x,y,s,n=30,fill=None): d.text((x,y),s,font=f(n),fill=fill or C['ink'])
def center(x,y,w,s,n=30,fill=None):
 b=d.textbbox((0,0),s,font=f(n));text(x+(w-b[2])/2,y,s,n,fill)
def box(x,y,w,h,fill='white',border=None,r=20):
 d.rounded_rectangle((x,y,x+w,y+h),radius=r,fill=fill,outline=border,width=2)
def arrow(points,color=None,width=4):
 import math
 color=color or C['blue'];d.line(points,fill=color,width=width,joint='curve')
 a,b=points[-2:];ang=math.atan2(b[1]-a[1],b[0]-a[0]);sz=13
 d.polygon([b,(b[0]-sz*math.cos(ang-.55),b[1]-sz*math.sin(ang-.55)),(b[0]-sz*math.cos(ang+.55),b[1]-sz*math.sin(ang+.55))],fill=color)
def label(x,y,s):
 box(x,y,62,38,C['blue'],r=9);center(x,y+4,62,s,22,'white')
def node(x,y,num,title,lines,w=385,h=218):
 box(x,y,w,h,C['pale'],C['line']);label(x+20,y+18,num);text(x+98,y+19,title,34)
 for i,s in enumerate(lines):text(x+24,y+78+i*36,s,28,C['muted'])
def section(x,y,n,title,sub):
 text(x,y,n,30,C['blue']);text(x+58,y-5,title,43);text(x,y+57,sub,27,C['muted'])

text(70,42,'경미한 차량 충돌 탐지 시스템',66)
text(73,127,'프로젝트 워크플로우  |  학습 코드 · 평가 상태 · 웹 추론 구성',32,C['muted'])
box(1640,57,690,90,C['pale'],r=16)
text(1667,69,'학습 코드: 3D Inception  /  웹 적용: SlowFast R50',28)
text(1667,109,'서로 다른 모델 — 학습 설정과 성능을 혼용하지 않음',24,C['muted'])
d.line((70,186,2330,186),fill=C['line'],width=2)

section(80,225,'01','모델 학습 과정','현재 model/train.py에 구현된 커스텀 3D Inception 학습 흐름')
xs=[80,525,970];y1=340;y2=625
node(xs[0],y1,'1','학습 데이터',['영상 MP4 + 주석 TXT','차량 BBox · 사고 시작 프레임','충돌 A=1 / 비충돌=0'])
node(xs[1],y1,'2','입력 전처리',['BBox 중심 ROI crop + pad','30프레임 · 224×224','RGB 변환 및 정규화'])
node(xs[2],y1,'3','학습·검증 분할',['학습 80% / 검증 20% 설정','도메인·방향·시나리오 층화','분할 seed = 42'])
arrow([(465,447),(516,447)]);arrow([(910,447),(960,447)])
node(xs[2],y2,'4','학습 데이터 증강',['좌우 반전 · 색상/밝기 변화','회색조 · 블러 · 노이즈','시간·BBox 지터 / 검증 제외'])
node(xs[1],y2,'5','3D Inception 학습',['CrossEntropyLoss + Adam','학습률 0.00003 · batch 8','최대 100 epoch · 처음부터 학습'])
node(xs[0],y2,'6','검증·가중치 저장',['RC / 실제 영상 각각 검증','실제 영상 val loss 우선 감시','최저 손실 모델 가중치 저장'])
arrow([(1162,558),(1162,615)]);arrow([(960,734),(918,734)]);arrow([(515,734),(474,734)])
box(80,876,1275,88,C['pale'],r=14)
text(104,889,'학습 안정화',28,C['blue'])
text(290,889,'LR 감소: 정체 3회 시 ×0.5  ·  Gradient clipping: 1.0',28)
text(290,927,'Early stopping: patience 15  ·  CUDA 사용 시 AMP',26,C['muted'])
text(80,993,'로컬 데이터 확인',28,C['ink'])
text(80,1035,'label/: TXT 935개   |   기본 data/train/: MP4 0개',28,C['muted'])
text(80,1074,'파일 보유 수이며, 실제 학습에 사용된 표본 수는 확인되지 않았습니다.',25,C['muted'])

d.line((1400,230,1400,1110),fill=C['line'],width=2)
section(1450,225,'02','모델 성능','SlowFast의 정량 평가 결과: 이 폴더에서 확인되지 않음')
rows=[('정확도  Accuracy','미확인'),('재현율  Recall','미확인'),('정밀도  Precision','미확인'),('F1-score','미확인'),('오탐율  FP / (FP+TN)','미확인')]
for i,(name,value) in enumerate(rows):
 y=345+i*79
 text(1465,y,name,32)
 box(2155,y-3,150,49,C['op'],r=10);center(2155,y+3,150,value,27,C['orange'])
 d.line((1465,y+63,2305,y+63),fill=C['line'],width=2)
box(1450,777,880,166,C['pale'],r=16)
text(1475,795,'평가 코드에 구현된 지표',31)
text(1475,843,'영상 단위 이진 판정 → TP / FN / FP / TN 집계',27,C['muted'])
text(1475,884,'기존 eval 진입점은 3D Inception용 모델을 로드',27,C['muted'])
text(1450,975,'학습 곡선: 저장된 epoch별 로그 미확인',29)
text(1450,1020,'기본 data/eval/ 없음 → 현재 정량 평가 불가',28,C['muted'])
text(1450,1060,'예측 확률이나 가중치 파일명은 정확도가 아닙니다.',27,C['muted'])

d.line((70,1140,2330,1140),fill=C['line'],width=2)
section(80,1172,'03','웹 분석 워크플로우','앞서 웹에 연결한 SlowFast R50 · CPU 추론 · 이미 학습된 .pth 가중치 사용')
y=1300
box(80,y,280,255,C['pale'],C['line']);center(80,y+24,280,'영상 등록',35)
center(80,y+88,280,'업로드 / 가져오기',28);center(80,y+132,280,'대상 영상 선택',28)
center(80,y+198,280,'React 웹 화면',25,C['muted'])
arrow([(365,y+125),(411,y+125)])
box(420,y,360,255,C['pale'],C['line']);center(420,y+24,360,'차량 지정·전처리',35)
center(420,y+88,360,'YOLO11x 탐지 / BBox 지정',26)
center(420,y+132,360,'ROI · 30프레임 · 224×224',26)
center(420,y+198,360,'윈도 이동: 5프레임',25,C['muted'])
arrow([(785,y+125),(833,y+125)])
box(840,y,760,255,C['pale'],C['line']);center(840,y+18,760,'SlowFast R50  |  3D ResNet 기반',34)
box(865,y+77,323,60,'#E0ECFB',r=10);center(865,y+90,323,'Slow · 8프레임',29)
box(865,y+163,323,60,'#C7DDF8',r=10);center(865,y+176,323,'Fast · 32프레임',29)
arrow([(1030,y+162),(1030,y+141)],C['blue'],3)
arrow([(1196,y+106),(1220,y+106),(1220,y+148),(1241,y+148)])
arrow([(1196,y+193),(1220,y+193),(1220,y+148)],C['blue'],3)
text(1260,y+89,'특징 결합',29);text(1260,y+132,'2304 → 2 클래스',28)
text(1260,y+178,'정상 / 충돌 확률',28)
arrow([(1605,y+125),(1653,y+125)])
box(1660,y,670,255,C['gp'],C['line']);center(1660,y+24,670,'사고 의심 구간·CAM 결과',35)
center(1660,y+88,670,'확률 + 움직임 + 지속시간 조건으로 필터',28)
center(1660,y+132,670,'구간·시점 추출 → CAM 오버레이 클립',28)
center(1660,y+198,670,'후보 결과를 수사관이 최종 확인',26,C['green'])
box(80,1590,2250,100,C['pale'],r=14)
text(107,1605,'서비스 연결',29,C['blue']);text(312,1605,'React → FastAPI → Celery / Redis → SlowFast → MySQL·클립 저장 → 웹 결과',29)
text(312,1650,'적용 가중치: weights_2026-09-11/slowfast_r50_2026-09-11_best.pth',27,C['muted'])
text(80,1730,'추적 가능한 근거',28)
text(80,1775,'학습: dataset.py · train.py · config.py  /  평가: evaluate.py · main.py',25,C['muted'])
text(80,1812,'웹: prediction_job.py · slowfast_service.py · predict_cam.py · routers/videos.py',25,C['muted'])
text(1735,1812,'코드·파일 확인 기준: 2026-09-21',25,C['muted'])
im.save(OUT/'project_workflow.png',dpi=(200,200))
print(OUT/'project_workflow.png')
