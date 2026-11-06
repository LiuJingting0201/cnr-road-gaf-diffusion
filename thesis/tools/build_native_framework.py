"""Vector diagram of the implemented G33 synthesis and terminal feedback path."""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor


destination = Path(__file__).resolve().parents[1]/"figures/g33-framework.pdf"
c = canvas.Canvas(str(destination), pagesize=(460, 330))
c.setTitle("Complete-trip diffusion and terminal GAF feedback")
ink, blue, green = HexColor("#243b50"), HexColor("#eaf2f9"), HexColor("#e8f3ef")


def box(x, y, w, h, lines, color=blue):
    c.setFillColor(color); c.setStrokeColor(ink)
    c.roundRect(x, y, w, h, 5, fill=1, stroke=1)
    c.setFillColor(ink)
    for i, line in enumerate(lines):
        c.setFont("Helvetica-Bold" if i == 0 else "Helvetica", 9)
        c.drawCentredString(x+w/2, y+h/2+(len(lines)-1)*5-i*11-3, line)


def arrow(points):
    import math
    c.setStrokeColor(ink); c.setFillColor(ink); c.setLineWidth(.8)
    path = c.beginPath(); path.moveTo(*points[0])
    for p in points[1:]: path.lineTo(*p)
    c.drawPath(path)
    (x0, y0), (x, y) = points[-2:]
    angle = math.atan2(y-y0, x-x0)
    head = c.beginPath(); head.moveTo(x, y)
    for side in (-1, 1):
        head.lineTo(x-5*math.cos(angle)+side*2.2*math.sin(angle), y-5*math.sin(angle)-side*2.2*math.cos(angle))
    head.close(); c.drawPath(head, fill=1, stroke=0)


c.setFillColor(ink); c.setFont("Helvetica-Bold", 12)
c.drawString(15, 311, "Complete journeys first; image feedback second")
box(15, 237, 115, 51, ["Public request", "Route, class, endpoints"])
box(175, 257, 135, 31, ["Train-side duration prior", "Sample trip time T"])
box(330, 257, 115, 31, ["Initial Gaussian noise"])
box(175, 196, 270, 40, ["Conditional motion diffusion", "Route / class / T; masked physical-time sequence"])
arrow([(130, 262), (151, 262), (151, 216), (175, 216)])
arrow([(151, 262), (151, 272), (175, 272)])
arrow([(242, 257), (242, 236)])
arrow([(387, 257), (387, 236)])
box(175, 137, 270, 40, ["Progress and road decoding", "Nonnegative increments, exact route, endpoint arrival"])
arrow([(310, 196), (310, 177)])
arrow([(72, 237), (72, 157), (175, 157)])
c.setFont("Helvetica", 8); c.drawString(80, 165, "Route geometry")
box(15, 56, 115, 46, ["K = 8 candidates", "time, id, x, y, type"])
arrow([(175, 147), (72, 147), (72, 102)])
box(175, 78, 118, 34, ["Spatial image + CNN", "Inherited Hilbert encoding"], green)
box(175, 26, 118, 34, ["Velocity image + CNN", "Speed derived from positions"], green)
arrow([(130, 79), (150, 79), (150, 95), (175, 95)])
arrow([(150, 79), (150, 43), (175, 43)])
box(330, 51, 115, 56, ["Terminal selection", "Uniform / S / V / SV", "One candidate per request"], green)
arrow([(293, 95), (330, 95)])
arrow([(293, 43), (310, 43), (310, 65), (330, 65)])
arrow([(388, 51), (388, 28)])
c.setFont("Helvetica-Bold", 9); c.drawCentredString(388, 15, "Synthetic trajectory data")
c.showPage(); c.save()
print(destination)
