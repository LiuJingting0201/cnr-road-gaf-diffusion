"""One-page Chinese handout accompanying the English thesis."""
from pathlib import Path
import sys
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
import pypdfium2 as pdfium
s=Path(sys.argv[1]);pdfmetrics.registerFont(TTFont('Hei','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0))
c=canvas.Canvas(str(s),pagesize=(595,842));c.setTitle('同一条道路上，公交和私家车怎样走？')
ink='#193741';gray='#4b626b';accent='#00796f'
def txt(x,y,st,size=11,col=ink):
    c.setFont('Hei',size);c.setFillColor(HexColor(col));c.drawString(x,y,st)
def rule(y):
    c.setStrokeColor(HexColor('#cdd8da'));c.line(35,y,560,y)
txt(35,800,'硕士论文研究总览     /     路网条件扩散轨迹合成',10,accent)
txt(35,758,'同一条道路上，公交和私家车怎样走？',25)
txt(35,727,'给定路网、起点和参与者类别，输出 512 个带时间的坐标。',13)
txt(35,705,'路径规定可走的道路；扩散模型生成沿路的运动；Hilbert–GAF 给候选打分。',10.2,gray)
rule(685)
doc=pdfium.PdfDocument('thesis/figures/shared-road-case.pdf')
im=doc[0].render(scale=1.75).to_pil()
c.drawImage(ImageReader(im),35,337,width=525,height=525*324/490)
txt(35,326,'固定首例：同一路、同起点、同噪声。公交走 1,182 米，私家车走 1,515 米。',10,gray)
rule(310)
for y,head,a,b in [
 (283,'路网可行性','53 个严格留出条件，1,272 条扩散候选全部通过道路图检查；','同解码器的经验基线也全部通过。道路一致性主要由路径结构提供。'),
 (210,'可控的运动','八批新仿真的 96 条共享路径上，只换成公交标签，平均少走 343.5 米，','近乎静止时间占比高 11.7 个百分点。这个结果衡量模型对类别输入的响应。'),
 (137,'GAF 的作用','完整轨迹分类与同校研究处于相近量级；自训评分器从同批候选中选出的轨迹，','平均速度更接近参考，但整批数据质量和下游增广没有稳定提高。')]:
    txt(35,y,head,14,accent);txt(143,y,a,10.2);txt(143,y-18,b,10.2,gray);rule(y-37)
txt(35,59,'适用范围',10,accent);txt(93,59,'同一张 MoST 路网、受控 SUMO 需求、五类移动子集。',10,gray)
txt(35,37,'单车生成；新仿真没有原公交时刻表。模型类别差异不能直接解释为真实公交与汽车的因果差异。',9,gray)
c.save()
