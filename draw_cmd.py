from arith import *
from sympy import *

#-------------------------
# Geometory
#-------------------------

# this design is not "open" for extension
def extract_sym(cmd):
    if cmd['type'] == 'lineseg':
        x0, y0 = cmd["start"]
        x1, y1 = cmd["end"]
        return (x0, y0, x1, y1)
    elif cmd['type'] == 'circle':
        x,y = cmd["origin"]
        r = cmd["r"]
        return (x, y, r)
    else:
        return ()

def eval_cmd(cmd, v):
    ret = dict(**cmd)
    if cmd['type'] == 'lineseg':
        ret["start"] = (v[0], v[1])
        ret["end"] = (v[2], v[3])
        return ret
    elif cmd['type'] == 'circle':
        ret["origin"] = (v[0], v[1])
        ret["r"] = v[2]
        return ret
    return ret

def gen_eval_draw_cmd_f(cmd, allsyms, context):
    eval_draw_param = lambdify(allsyms + list(context.keys()), extract_sym(cmd))
    def draw_cmds_f(q, dq, other):
        return eval_cmd(cmd, eval_draw_param(*q, *dq, *other, *list(context.values())))
    return draw_cmds_f

def draw_circle_cmd(p, r, color=None, width=1, name="-"):
    return [{"type": "circle", "color":color, "origin":(p[0], p[1]), "r":r, "width":width, "name":name}]

def draw_lineseg_cmd(p0, p1, color=None, width=1, name="-"):
    return [{"type": "lineseg", "color":color, "start":(p0[0], p0[1]), "end":(p1[0], p1[1]), "width":width, "name":name}]

def draw_lineseg_turtle_cmd(X, l, color=None, width=1, name="-"):
    s, c, x, y = Xtoscxy(X)
    dir = Matrix([c,s])
    org = Matrix([x,y])
    return draw_lineseg_cmd(org, org+dir*l, color, width, name)

def draw_dir_circle_cmd(X, r, color=None, width=1, name="-"):
    s, c, x, y = Xtoscxy(X)
    dir = Matrix([c,s])
    org = Matrix([x,y])
    radius = draw_lineseg_cmd(org, org+dir*r, color, width, name) 
    return radius + draw_circle_cmd((x,y), r, color, width, name)

def plot_points_cmd(ps, r, color=None, width=1, name="-"):
    return sum([draw_circle_cmd(p, r, color, width, name) for p in ps], [])

def plot_points_with_line_cmd(ps, color=None, width=1, name="-"):
    ret = []
    for p0, p1 in zip(ps[0:-1], ps[1:]):
        ret = ret + draw_lineseg_cmd(p0, p1, color, width, name)
    return ret

