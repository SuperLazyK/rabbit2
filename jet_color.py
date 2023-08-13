
def interpolate(val, y0, x0, y1, x1):
    return (val-x0)*(y1-y0)/(x1-x0) + y0

def jet_elm(val):
    if val <= -0.75:
       return 0;
    elif val <= -0.25:
         return interpolate( val, 0.0, -0.75, 1.0, -0.25 );
    elif val <= 0.25:
         return 1.0;
    elif val <= 0.75:
         return interpolate( val, 1.0, 0.25, 0.0, 0.75 );
    else:
         return 0.0

def jet(v): # v = 0-1 -> RGB
    return (int(255*jet_elm(v - 0.5)), int(255*jet_elm(v)), int(255*jet_elm(v+0.5)))

