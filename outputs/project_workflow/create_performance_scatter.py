from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
O=Path(__file__).resolve().parent
im=Image.new('RGB',(1800,1250),'white');d=ImageDraw.Draw(im)
font='/System/Library/Fonts/AppleSDGothicNeo.ttc'
def t(x,y,s,n=30,c='#263C52'): d.text((x,y),s,font=ImageFont.truetype(font,n),fill=c)
left,right,top,bottom=210,1640,225,1000
X=lambda x:left+(x-75)/30*(right-left)
Y=lambda y:bottom-y/1.1*(bottom-top)
t(120,55,'모델별 검증 성능 · 점그래프',54)
t(123,133,'검증 정확도와 사고 F1의 관계',31,'#62758A')
for v in [0,.25,.5,.75,1]:
 y=Y(v);d.line((left,y,right,y),fill='#E3EAF2',width=2);t(125,y-19,f'{v:.2f}',27)
for v in [75,80,85,90,95,100,105]:
 x=X(v);d.line((x,top,x,bottom),fill='#E3EAF2',width=2);t(x-22,bottom+22,str(v),27)
d.line((left,top,left,bottom,right,bottom),fill='#52677B',width=3)
t(110,183,'사고 F1',30);t(1320,1070,'검증 정확도 (%)',32)
points=[('X3D-S',81.8,.50),('S3D',90.9,.75),('SlowFast-R50',100,1)]
# Least-squares fit of provided model-level aggregate observations, not training samples.
xs=[p[1] for p in points];ys=[p[2] for p in points];mx=sum(xs)/3;my=sum(ys)/3
slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs);intercept=my-slope*mx
x1,x2=X(min(xs)),X(max(xs));y1,y2=Y(slope*min(xs)+intercept),Y(slope*max(xs)+intercept)
for i in range(0,100,3):
 a=i/100;b=min((i+1.6)/100,1);d.line((x1+(x2-x1)*a,y1+(y2-y1)*a,x1+(x2-x1)*b,y1+(y2-y1)*b),fill='#7198CC',width=4)
for name,x,y in points:
 px,py=X(x),Y(y);c='#246FD0' if name=='SlowFast-R50' else '#698AAF'
 d.ellipse((px-12,py-12,px+12,py+12),fill=c)
 dx,dy=(23,20) if name=='SlowFast-R50' else (23,-105)
 if name=='SlowFast-R50':dx=23;dy=20
 t(px+dx,py+dy,name,33,c);t(px+dx,py+dy+44,f'{x:g}%  /  F1 {y:.2f}',28)
d.line((1150,125,1220,125),fill='#7198CC',width=4);t(1240,105,'3개 모델의 선형 추세',28)
t(123,1150,'사용자 제공 검증 결과 3개 점 · 개별 학습 샘플이나 epoch별 학습 곡선이 아님',28,'#62758A')
im.save(O/'validation_performance_scatter.png',dpi=(200,200))
