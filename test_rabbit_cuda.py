from linktree import *
from jointlink import *
from simulation import *
from draw_cmd import *
from sympy import pi, symbols, Pow, Function
import cupy as cp
import time

kernel_update = cp.ElementwiseKernel(
        in_params='float64 x, float64 y, float64 qa, float64 q0, float64 q1, float64 q2, float64 dx, float64 dy, float64 dqa, float64 dq0, float64 dq1, float64 dq2, float64 u0, float64 u1, float64 u2, float64 dt',
        out_params='float64 new_x, float64 new_y, float64 new_qa, float64 new_q0, float64 new_q1, float64 new_q2, float64 new_dx, float64 new_dy, float64 new_dqa, float64 new_dq0, float64 new_dq1, float64 new_dq2',
        operation=\
        '''
float cf0 = 0;
float cf1 = 0;
float cf2 = 0;
float K = 10000;
float Dt = 1./dt;
float mu = 1;
float d = 0;

float ca = cos(qa);
float sa = sin(qa);

if (y > 0) {
    if (abs(ca) < 0.05) {
        if (ca > 0) {
            d = 100000;
        } else {
            d = -100000;
        }
    } else {
        d = y / ca;
    }
}
float z0 = 0.55;

float cpx = 0;
float cpy = 0;
float fx = 0;
float fy = 0;

if (d <= z0 && d >= 0) {
    cpx = x + d * sa;
    float vx = dx + y/(ca*ca)*dqa + dy * sa/ca;
    float f_spring_x = K * (z0 - d) * sa;
    float f_spring_y = fmaxf(0, K * (z0 - d) * ca);
    float f_fric_x   = fminf(fmaxf(-Dt * vx, -mu * f_spring_y), mu * f_spring_y);
    fx = f_spring_x + f_fric_x;
    fy = f_spring_y;
}

float H00 = 87;
float H01 = 0;
float H02 = (55.0*sin(q0 + q1) + 20.0*sin(q0 + q1 + q2) + 60.0*sin(q0))*sin(qa)/2 - (55.0*cos(q0 + q1) + 20.0*cos(q0 + q1 + q2) + 60.0*cos(q0) + 5.81)*cos(qa)/2;
float H03 = (55.0*sin(q0 + q1) + 20.0*sin(q0 + q1 + q2) + 60.0*sin(q0))*sin(qa)/2 - (55.0*cos(q0 + q1) + 20.0*cos(q0 + q1 + q2) + 60.0*cos(q0))*cos(qa)/2;
float H04 = (55.0*sin(q0 + q1) + 20.0*sin(q0 + q1 + q2))*sin(qa)/2 - (55.0*cos(q0 + q1) + 20.0*cos(q0 + q1 + q2))*cos(qa)/2;
float H05 = -10.0*cos(q0 + q1 + q2 + qa);
float H10 = 0;
float H11 = 87;
float H12 = -30.0*sin(q0 + qa) - 27.5*sin(q0 + q1 + qa) - 10.0*sin(q0 + q1 + q2 + qa) - 2.905*sin(qa);
float H13 = -30.0*sin(q0 + qa) - 27.5*sin(q0 + q1 + qa) - 10.0*sin(q0 + q1 + q2 + qa);
float H14 = -27.5*sin(q0 + q1 + qa) - 10.0*sin(q0 + q1 + q2 + qa);
float H15 = -10.0*sin(q0 + q1 + q2 + qa);
float H20 = (55.0*sin(q0 + q1) + 20.0*sin(q0 + q1 + q2) + 60.0*sin(q0))*sin(qa)/2 - (55.0*cos(q0 + q1) + 20.0*cos(q0 + q1 + q2) + 60.0*cos(q0) + 5.81)*cos(qa)/2;
float H21 = -30.0*sin(q0 + qa) - 27.5*sin(q0 + q1 + qa) - 10.0*sin(q0 + q1 + q2 + qa) - 2.905*sin(qa);
float H22 = 8.0*cos(q1 + q2) + 22.0*cos(q1) + 10.0*cos(q2) + 27.180575;
float H23 = 8.0*cos(q1 + q2) + 22.0*cos(q1) + 10.0*cos(q2) + 25.975;
float H24 = 4.0*cos(q1 + q2) + 11.0*cos(q1) + 10.0*cos(q2) + 14.375;
float H25 = 4.0*cos(q1 + q2) + 5.0*cos(q2) + 2.5;
float H30 = (55.0*sin(q0 + q1) + 20.0*sin(q0 + q1 + q2) + 60.0*sin(q0))*sin(qa)/2 - (55.0*cos(q0 + q1) + 20.0*cos(q0 + q1 + q2) + 60.0*cos(q0))*cos(qa)/2;
float H31 = -30.0*sin(q0 + qa) - 27.5*sin(q0 + q1 + qa) - 10.0*sin(q0 + q1 + q2 + qa);
float H32 = 8.0*cos(q1 + q2) + 22.0*cos(q1) + 10.0*cos(q2) + 25.975;
float H33 = 8.0*cos(q1 + q2) + 22.0*cos(q1) + 10.0*cos(q2) + 25.975;
float H34 = 4.0*cos(q1 + q2) + 11.0*cos(q1) + 10.0*cos(q2) + 14.375;
float H35 = 4.0*cos(q1 + q2) + 5.0*cos(q2) + 2.5;
float H40 = (55.0*sin(q0 + q1) + 20.0*sin(q0 + q1 + q2))*sin(qa)/2 - (55.0*cos(q0 + q1) + 20.0*cos(q0 + q1 + q2))*cos(qa)/2;
float H41 = -27.5*sin(q0 + q1 + qa) - 10.0*sin(q0 + q1 + q2 + qa);
float H42 = 4.0*cos(q1 + q2) + 11.0*cos(q1) + 10.0*cos(q2) + 14.375;
float H43 = 4.0*cos(q1 + q2) + 11.0*cos(q1) + 10.0*cos(q2) + 14.375;
float H44 = 10.0*cos(q2) + 14.375;
float H45 = 5.0*cos(q2) + 2.5;
float H50 = -10.0*cos(q0 + q1 + q2 + qa);
float H51 = -10.0*sin(q0 + q1 + q2 + qa);
float H52 = 4.0*cos(q1 + q2) + 5.0*cos(q2) + 2.5;
float H53 = 4.0*cos(q1 + q2) + 5.0*cos(q2) + 2.5;
float H54 = 5.0*cos(q2) + 2.5;
float H55 = 2.50000000000000;

float rhs0 = fx - 55.0*dq0*dq1*sin(q0 + q1 + qa) - 20.0*dq0*dq1*sin(q0 + q1 + q2 + qa) - 20.0*dq0*dq2*sin(q0 + q1 + q2 + qa) - 60.0*dq0*dqa*sin(q0 + qa) - 55.0*dq0*dqa*sin(q0 + q1 + qa) - 20.0*dq0*dqa*sin(q0 + q1 + q2 + qa) - 20.0*dq1*dq2*sin(q0 + q1 + q2 + qa) - 55.0*dq1*dqa*sin(q0 + q1 + qa) - 20.0*dq1*dqa*sin(q0 + q1 + q2 + qa) - 20.0*dq2*dqa*sin(q0 + q1 + q2 + qa) - 30.0*powf(dq0, 2)*sin(q0 + qa) - 27.5*powf(dq0, 2)*sin(q0 + q1 + qa) - 10.0*powf(dq0, 2)*sin(q0 + q1 + q2 + qa) - 27.5*powf(dq1, 2)*sin(q0 + q1 + qa) - 10.0*powf(dq1, 2)*sin(q0 + q1 + q2 + qa) - 10.0*powf(dq2, 2)*sin(q0 + q1 + q2 + qa) - 30.0*powf(dqa, 2)*sin(q0 + qa) - 27.5*powf(dqa, 2)*sin(q0 + q1 + qa) - 10.0*powf(dqa, 2)*sin(q0 + q1 + q2 + qa) - 2.905*powf(dqa, 2)*sin(qa);
float rhs1 = fy - 20.0*dq0*dq1*sin(q0 + q1)*sin(qa)*cos(q2) - 20.0*dq0*dq1*sin(q1 + qa)*sin(q2)*cos(q0) - 20.0*dq0*dq1*sin(q0)*sin(q1)*cos(q2 + qa) + 20.0*dq0*dq1*cos(q0 + q2)*cos(q1)*cos(qa) + 55.0*dq0*dq1*cos(q0 + q1 + qa) - 20.0*dq0*dq2*sin(q0 + q1)*sin(qa)*cos(q2) - 20.0*dq0*dq2*sin(q1 + qa)*sin(q2)*cos(q0) - 20.0*dq0*dq2*sin(q0)*sin(q1)*cos(q2 + qa) + 20.0*dq0*dq2*cos(q0 + q2)*cos(q1)*cos(qa) - 20.0*dq0*dqa*sin(q0 + q1)*sin(qa)*cos(q2) - 20.0*dq0*dqa*sin(q1 + qa)*sin(q2)*cos(q0) - 20.0*dq0*dqa*sin(q0)*sin(q1)*cos(q2 + qa) + 20.0*dq0*dqa*cos(q0 + q2)*cos(q1)*cos(qa) + 60.0*dq0*dqa*cos(q0 + qa) + 55.0*dq0*dqa*cos(q0 + q1 + qa) - 20.0*dq1*dq2*sin(q0 + q1)*sin(qa)*cos(q2) - 20.0*dq1*dq2*sin(q1 + qa)*sin(q2)*cos(q0) - 20.0*dq1*dq2*sin(q0)*sin(q1)*cos(q2 + qa) + 20.0*dq1*dq2*cos(q0 + q2)*cos(q1)*cos(qa) - 20.0*dq1*dqa*sin(q0 + q1)*sin(qa)*cos(q2) - 20.0*dq1*dqa*sin(q1 + qa)*sin(q2)*cos(q0) - 20.0*dq1*dqa*sin(q0)*sin(q1)*cos(q2 + qa) + 20.0*dq1*dqa*cos(q0 + q2)*cos(q1)*cos(qa) + 55.0*dq1*dqa*cos(q0 + q1 + qa) - 20.0*dq2*dqa*sin(q0 + q1)*sin(qa)*cos(q2) - 20.0*dq2*dqa*sin(q1 + qa)*sin(q2)*cos(q0) - 20.0*dq2*dqa*sin(q0)*sin(q1)*cos(q2 + qa) + 20.0*dq2*dqa*cos(q0 + q2)*cos(q1)*cos(qa) - 10.0*powf(dq0, 2)*sin(q0 + q1)*sin(qa)*cos(q2) - 10.0*powf(dq0, 2)*sin(q1 + qa)*sin(q2)*cos(q0) - 10.0*powf(dq0, 2)*sin(q0)*sin(q1)*cos(q2 + qa) + 10.0*powf(dq0, 2)*cos(q0 + q2)*cos(q1)*cos(qa) + 30.0*powf(dq0, 2)*cos(q0 + qa) + 27.5*powf(dq0, 2)*cos(q0 + q1 + qa) - 10.0*powf(dq1, 2)*sin(q0 + q1)*sin(qa)*cos(q2) - 10.0*powf(dq1, 2)*sin(q1 + qa)*sin(q2)*cos(q0) - 10.0*powf(dq1, 2)*sin(q0)*sin(q1)*cos(q2 + qa) + 10.0*powf(dq1, 2)*cos(q0 + q2)*cos(q1)*cos(qa) + 27.5*powf(dq1, 2)*cos(q0 + q1 + qa) - 10.0*powf(dq2, 2)*sin(q0 + q1)*sin(qa)*cos(q2) - 10.0*powf(dq2, 2)*sin(q1 + qa)*sin(q2)*cos(q0) - 10.0*powf(dq2, 2)*sin(q0)*sin(q1)*cos(q2 + qa) + 10.0*powf(dq2, 2)*cos(q0 + q2)*cos(q1)*cos(qa) - 10.0*powf(dqa, 2)*sin(q0 + q1)*sin(qa)*cos(q2) - 10.0*powf(dqa, 2)*sin(q1 + qa)*sin(q2)*cos(q0) - 10.0*powf(dqa, 2)*sin(q0)*sin(q1)*cos(q2 + qa) + 10.0*powf(dqa, 2)*cos(q0 + q2)*cos(q1)*cos(qa) + 30.0*powf(dqa, 2)*cos(q0 + qa) + 27.5*powf(dqa, 2)*cos(q0 + q1 + qa) + 2.905*powf(dqa, 2)*cos(qa) - 853.47;
float rhs2 = cpx*fy - cpy*fx + fx*y - fy*x + 8.0*dq0*dq1*sin(q1 + q2) + 22.0*dq0*dq1*sin(q1) + 8.0*dq0*dq2*sin(q1 + q2) + 10.0*dq0*dq2*sin(q2) + 8.0*dq1*dq2*sin(q1 + q2) + 10.0*dq1*dq2*sin(q2) + 8.0*dq1*dqa*sin(q1 + q2) + 22.0*dq1*dqa*sin(q1) + 8.0*dq2*dqa*sin(q1 + q2) + 10.0*dq2*dqa*sin(q2) + 4.0*powf(dq1, 2)*sin(q1 + q2) + 11.0*powf(dq1, 2)*sin(q1) + 4.0*powf(dq2, 2)*sin(q1 + q2) + 5.0*powf(dq2, 2)*sin(q2) + 294.3*sin(q0 + qa) + 269.775*sin(q0 + q1 + qa) + 98.1*sin(q0 + q1 + q2 + qa) + 28.49805*sin(qa);
float rhs3 = cf0 + u0 + 8.0*dq0*dq1*sin(q1 + q2) + 22.0*dq0*dq1*sin(q1) + 8.0*dq0*dq2*sin(q1 + q2) + 10.0*dq0*dq2*sin(q2) + 8.0*dq1*dq2*sin(q1 + q2) + 10.0*dq1*dq2*sin(q2) + 8.0*dq1*dqa*sin(q1 + q2) + 22.0*dq1*dqa*sin(q1) + 8.0*dq2*dqa*sin(q1 + q2) + 10.0*dq2*dqa*sin(q2) + 4.0*powf(dq1, 2)*sin(q1 + q2) + 11.0*powf(dq1, 2)*sin(q1) + 4.0*powf(dq2, 2)*sin(q1 + q2) + 5.0*powf(dq2, 2)*sin(q2) + 294.3*sin(q0 + qa) + 269.775*sin(q0 + q1 + qa) + 98.1*sin(q0 + q1 + q2 + qa);
float rhs4 = cf1 + u1 + 10.0*dq0*dq2*sin(q2) - 8.0*dq0*dqa*sin(q1 + q2) - 22.0*dq0*dqa*sin(q1) + 10.0*dq1*dq2*sin(q2) + 10.0*dq2*dqa*sin(q2) - 4.0*powf(dq0, 2)*sin(q1 + q2) - 11.0*powf(dq0, 2)*sin(q1) + 5.0*powf(dq2, 2)*sin(q2) - 4.0*powf(dqa, 2)*sin(q1 + q2) - 11.0*powf(dqa, 2)*sin(q1) + 269.775*sin(q0 + q1 + qa) + 98.1*sin(q0 + q1 + q2 + qa);
float rhs5 = cf2 + u2 - 10.0*dq0*dq1*sin(q2) - 8.0*dq0*dqa*sin(q1 + q2) - 10.0*dq0*dqa*sin(q2) - 10.0*dq1*dqa*sin(q2) - 4.0*powf(dq0, 2)*sin(q1 + q2) - 5.0*powf(dq0, 2)*sin(q2) - 5.0*powf(dq1, 2)*sin(q2) - 4.0*powf(dqa, 2)*sin(q1 + q2) - 5.0*powf(dqa, 2)*sin(q2) + 98.1*sin(q0 + q1 + q2 + qa);

float redH22 = H22-(H12*H21)/H11-(H02*H20)/H00;
float redH23 = H23-(H13*H21)/H11-(H03*H20)/H00;
float redH24 = H24-(H14*H21)/H11-(H04*H20)/H00;
float redH25 = H25-(H15*H21)/H11-(H05*H20)/H00;
float rhsred2 = rhs2-(H21*rhs1)/H11-(H20*rhs0)/H00;

float redH32 = H32-(H12*H31)/H11-(H02*H30)/H00;
float redH33 = H33-(H13*H31)/H11-(H03*H30)/H00;
float redH34 = H34-(H14*H31)/H11-(H04*H30)/H00;
float redH35 = H35-(H15*H31)/H11-(H05*H30)/H00;
float rhsred3 = rhs3-(H31*rhs1)/H11-(H30*rhs0)/H00;

float redH42 = H42-(H12*H41)/H11-(H02*H40)/H00;
float redH43 = H43-(H13*H41)/H11-(H03*H40)/H00;
float redH44 = H44-(H14*H41)/H11-(H04*H40)/H00;
float redH45 = H45-(H15*H41)/H11-(H05*H40)/H00;
float rhsred4 = rhs4-(H41*rhs1)/H11-(H40*rhs0)/H00;

float redH52 = H52-(H12*H51)/H11-(H02*H50)/H00;
float redH53 = H53-(H13*H51)/H11-(H03*H50)/H00;
float redH54 = H54-(H14*H51)/H11-(H04*H50)/H00;
float redH55 = H55-(H15*H51)/H11-(H05*H50)/H00;
float rhsred5 = rhs5-(H51*rhs1)/H11-(H50*rhs0)/H00;

float adjH22 = redH33*(redH44*redH55-redH45*redH54)-redH34*(redH43*redH55-redH45*redH53)+redH35*(redH43*redH54-redH44*redH53)   ;
float adjH23 = (-redH23*(redH44*redH55-redH45*redH54))+redH24*(redH43*redH55-redH45*redH53)-redH25*(redH43*redH54-redH44*redH53);
float adjH24 = redH23*(redH34*redH55-redH35*redH54)-redH24*(redH33*redH55-redH35*redH53)+redH25*(redH33*redH54-redH34*redH53)   ;
float adjH25 = (-redH23*(redH34*redH45-redH35*redH44))+redH24*(redH33*redH45-redH35*redH43)-redH25*(redH33*redH44-redH34*redH43);
float adjH32 = (-redH32*(redH44*redH55-redH45*redH54))+redH34*(redH42*redH55-redH45*redH52)-redH35*(redH42*redH54-redH44*redH52);
float adjH33 = redH22*(redH44*redH55-redH45*redH54)-redH24*(redH42*redH55-redH45*redH52)+redH25*(redH42*redH54-redH44*redH52)   ;
float adjH34 = (-redH22*(redH34*redH55-redH35*redH54))+redH24*(redH32*redH55-redH35*redH52)-redH25*(redH32*redH54-redH34*redH52);
float adjH35 = redH22*(redH34*redH45-redH35*redH44)-redH24*(redH32*redH45-redH35*redH42)+redH25*(redH32*redH44-redH34*redH42)   ;
float adjH42 = redH32*(redH43*redH55-redH45*redH53)-redH33*(redH42*redH55-redH45*redH52)+redH35*(redH42*redH53-redH43*redH52)   ;
float adjH43 = (-redH22*(redH43*redH55-redH45*redH53))+redH23*(redH42*redH55-redH45*redH52)-redH25*(redH42*redH53-redH43*redH52);
float adjH44 = redH22*(redH33*redH55-redH35*redH53)-redH23*(redH32*redH55-redH35*redH52)+redH25*(redH32*redH53-redH33*redH52)   ;
float adjH45 = (-redH22*(redH33*redH45-redH35*redH43))+redH23*(redH32*redH45-redH35*redH42)-redH25*(redH32*redH43-redH33*redH42);
float adjH52 = (-redH32*(redH43*redH54-redH44*redH53))+redH33*(redH42*redH54-redH44*redH52)-redH34*(redH42*redH53-redH43*redH52);
float adjH53 = redH22*(redH43*redH54-redH44*redH53)-redH23*(redH42*redH54-redH44*redH52)+redH24*(redH42*redH53-redH43*redH52)   ;
float adjH54 = (-redH22*(redH33*redH54-redH34*redH53))+redH23*(redH32*redH54-redH34*redH52)-redH24*(redH32*redH53-redH33*redH52);
float adjH55 = redH22*(redH33*redH44-redH34*redH43)-redH23*(redH32*redH44-redH34*redH42)+redH24*(redH32*redH43-redH33*redH42)   ;
float detredH = redH22*(redH33*(redH44*redH55-redH45*redH54)-redH34*(redH43*redH55-redH45*redH53)+redH35*(redH43*redH54-redH44*redH53))-redH23*(redH32*(redH44*redH55-redH45*redH54)-redH34*(redH42*redH55-redH45*redH52)+redH35*(redH42*redH54-redH44*redH52))+redH24*(redH32*(redH43*redH55-redH45*redH53)-redH33*(redH42*redH55-redH45*redH52)+redH35*(redH42*redH53-redH43*redH52))-redH25*(redH32*(redH43*redH54-redH44*redH53)-redH33*(redH42*redH54-redH44*redH52)+redH34*(redH42*redH53-redH43*redH52));
float rdetredH = 1. / detredH;

float ddqa = (adjH22 * rhsred2 + adjH23 * rhsred3 + adjH24 * rhsred4 + adjH25 * rhsred5) * rdetredH;
float ddq0 = (adjH32 * rhsred2 + adjH33 * rhsred3 + adjH34 * rhsred4 + adjH35 * rhsred5) * rdetredH;
float ddq1 = (adjH42 * rhsred2 + adjH43 * rhsred3 + adjH44 * rhsred4 + adjH45 * rhsred5) * rdetredH;
float ddq2 = (adjH52 * rhsred2 + adjH53 * rhsred3 + adjH54 * rhsred4 + adjH55 * rhsred5) * rdetredH;

float ddx  = (rhs0 - (H02 * ddqa + H03 * ddq0 + H04 * ddq1 + H05 * ddq2))/H00;
float ddy  = (rhs1 - (H12 * ddqa + H13 * ddq0 + H14 * ddq1 + H15 * ddq2))/H11;

new_dx  = dx  + ddx * dt;
new_dy  = dy  + ddy * dt;
new_dqa = dqa + ddqa * dt;
new_dq0 = dq0 + ddq0 * dt;
new_dq1 = dq1 + ddq1 * dt;
new_dq2 = dq2 + ddq2 * dt;
new_x  = x  + new_dx * dt;
new_y  = y  + new_dy * dt;
new_qa = qa + new_dqa * dt;
new_q0 = q0 + new_dq0 * dt;
new_q1 = q1 + new_dq1 * dt;
new_q2 = q2 + new_dq2 * dt;
''', name='update')


kernel_cost = cp.ElementwiseKernel(
        in_params='float64 x, float64 y, float64 qa, float64 q0, float64 q1, float64 q2, float64 dx, float64 dy, float64 dqa, float64 dq0, float64 dq1, float64 dq2, float64 u0, float64 u1, float64 u2, float64 c',
        out_params='float64 new_c',
        operation=\
        '''
new_c = c + 0;
''', name='cost')


cu_fx  = cp.ndarray((1,), np.float64)
cu_fy  = cp.ndarray((1,), np.float64)
cu_cpx = cp.ndarray((1,), np.float64)
cu_cpy = cp.ndarray((1,), np.float64)
cu_x   = cp.ndarray((1,), np.float64)
cu_y   = cp.ndarray((1,), np.float64)
cu_qa  = cp.ndarray((1,), np.float64)
cu_q0  = cp.ndarray((1,), np.float64)
cu_q1  = cp.ndarray((1,), np.float64)
cu_q2  = cp.ndarray((1,), np.float64)
cu_dx  = cp.ndarray((1,), np.float64)
cu_dy  = cp.ndarray((1,), np.float64)
cu_dqa = cp.ndarray((1,), np.float64)
cu_dq0 = cp.ndarray((1,), np.float64)
cu_dq1 = cp.ndarray((1,), np.float64)
cu_dq2 = cp.ndarray((1,), np.float64)

u0, u1, u2 = symbols('u0 u1 u2') # motor torq
fx, fy = symbols('fx fy') # friction force and normal force from ground
cf0, cf1, cf2 = symbols('cf0 cf1 cf2') # constraint force aligned axis
cpx, cpy = symbols('cpx cpy') # contact point
g = symbols('g')
powf = Function('powf')
l_stick, l_shin, l_thigh, l_ubody = symbols('lstick lshin lthigh lubody')
m_foot, m_stick, m_shin, m_thigh, m_ubody = symbols('mfoot mstick mshin mthigh mubody')

lstick= 0.83
lshin= 0.4
lthigh= 0.5
lubody= 0.5
mfoot= 5
mstick= 7
mshin= 10
mthigh= 30
mubody= 40
context = { 
        l_stick: lstick, l_shin: lshin, l_thigh: lthigh, l_ubody: lubody,
        m_foot: mfoot, m_stick: mstick, m_shin: mshin, m_thigh: mthigh,  m_ubody: mubody,
        g: 9.81
        }
z0 = 0.55


def contact_distance(x,y,qa):
    if y <= 0:
        return 0
    ca = np.cos(qa)
    if abs(ca) < 0.05:
        return np.inf if ca > 0 else -np.inf
    return y / ca


def contact_force(x,y,qa,q0,q1,q2,dx,dy,dqa,dq0,dq1,dq2,dt):
    K = 10000
    Dt = 1./dt
    mu = 1
    #mu = np.inf

    d = contact_distance(x,y,qa)
    if d <= z0 and d >= 0:
        ca = np.cos(qa)
        sa = np.sin(qa)
        pc = (x + d * sa, 0)
        vx = dx + y/(ca**2)*dqa + dy * sa/ca
        f_spring_x = K * (z0 - d) * sa
        f_spring_y = max(0, K * (z0 - d) * ca)
        f_fric_x   = np.clip(-Dt * vx, -mu * f_spring_y, mu * f_spring_y)
        fx = f_spring_x + f_fric_x
        fy = f_spring_y
        cpx = pc[0]
        cpy = pc[1]
    else:
        fx = 0
        fy = 0
        cpx = 0
        cpy = 0

    return fx,fy,cpx,cpy

def cu_update(x,y,qa,q0,q1,q2,dx,dy,dqa,dq0,dq1,dq2,u0,u1,u2,dt):
    cu_x   .set(np.array([x  ]))
    cu_y   .set(np.array([y  ]))
    cu_qa  .set(np.array([qa ]))
    cu_q0  .set(np.array([q0 ]))
    cu_q1  .set(np.array([q1 ]))
    cu_q2  .set(np.array([q2 ]))
    cu_dx  .set(np.array([dx ]))
    cu_dy  .set(np.array([dy ]))
    cu_dqa .set(np.array([dqa]))
    cu_dq0 .set(np.array([dq0]))
    cu_dq1 .set(np.array([dq1]))
    cu_dq2 .set(np.array([dq2]))
    kernel_update(cu_x,cu_y,cu_qa,cu_q0,cu_q1,cu_q2,cu_dx,cu_dy,cu_dqa,cu_dq0,cu_dq1,cu_dq2,u0, u1, u2, dt, cu_x,cu_y,cu_qa,cu_q0,cu_q1,cu_q2,cu_dx,cu_dy,cu_dqa,cu_dq0,cu_dq1,cu_dq2)
    return np.array([cp.asnumpy(cu_x)[0], cp.asnumpy(cu_y)[0] , cp.asnumpy(cu_qa)[0] , cp.asnumpy(cu_q0)[0] , cp.asnumpy(cu_q1)[0] , cp.asnumpy(cu_q2)[0]]), np.array([cp.asnumpy(cu_dx)[0] , cp.asnumpy(cu_dy)[0] , cp.asnumpy(cu_dqa)[0] , cp.asnumpy(cu_dq0)[0] , cp.asnumpy(cu_dq1)[0] , cp.asnumpy(cu_dq2)[0]])


def constraint_force(q0,q1,q2):
    cf0 = 0
    cf1 = 0
    cf2 = 0
    return cf0, cf1, cf2


def solve33(H00,H01,H02,H10,H11,H12,H20,H21,H22,rhs0,rhs1,rhs2):
    adjH00 = H11*H22-H12*H21
    adjH01 = H02*H21-H01*H22
    adjH02 = H01*H12-H02*H11
    adjH10 = H12*H20-H10*H22
    adjH11 = H00*H22-H02*H20
    adjH12 = H02*H10-H00*H12
    adjH20 = H10*H21-H11*H20
    adjH21 = H01*H20-H00*H21
    adjH22 = H00*H11-H01*H10
    detH   = H00*adjH00+H01*adjH10+H02*adjH20
    sol0 = (adjH00 * rhs0 + adjH01 * rhs1 + adjH02 * rhs2)/detH
    sol1 = (adjH10 * rhs0 + adjH11 * rhs1 + adjH12 * rhs2)/detH
    sol2 = (adjH20 * rhs0 + adjH21 * rhs1 + adjH22 * rhs2)/detH
    return sol0, sol1, sol2

class Rabbit(Simulator):
    def __init__(self):
        super().__init__()
        self.q_v = np.zeros(6)
        self.dq_v = np.zeros(6)
        self.reset_state()

    def reset_state(self):
        super().reset_state()
        self.u0 = 0
        self.u1 = 0
        self.u2 = 0
        self.cf0 = 0
        self.cf1 = 0
        self.cf2 = 0
        self.cpx = 0
        self.cpy = 0
        self.fx = 0
        self.fy = 0

        self.q_v[1] = 1
        self.q_v[3] = np.pi/6
        self.q_v[4] = -np.pi/3
        self.q_v[5] = np.pi/6
        self.message = [""]

    def sim_input_syms(self):
        return [u0, u1, u2, cf0, cf1, cf2, fx, fy, cpx, cpy]

    def sim_input_vals(self):
        return np.array([self.u0, self.u1, self.u2, self.cf0, self.cf1, self.cf2, self.fx, self.fy, self.cpx, self.cpy])

    def update(self, dt):
        x,y,qa,q0,q1,q2 = self.q_v[0],self.q_v[1],self.q_v[2],self.q_v[3],self.q_v[4],self.q_v[5]
        dx,dy,dqa,dq0,dq1,dq2 = self.dq_v[0],self.dq_v[1],self.dq_v[2],self.dq_v[3],self.dq_v[4],self.dq_v[5]
        self.q_v,self.dq_v = cu_update(x,y,qa,q0,q1,q2,dx,dy,dqa,dq0,dq1,dq2,self.u0,self.u1,self.u2,dt)
        self.t = self.t + dt

    def draw_text(self):
        return super().draw_text() + self.message

    def draw(self):
        x,y,qa,q0,q1,q2 = self.q_v[0],self.q_v[1],self.q_v[2],self.q_v[3],self.q_v[4],self.q_v[5]

        pbase = np.array([x, y])
        pknee = pbase + lshin * np.array([-sin(q0 + qa), cos(q0 + qa)])
        pwest = pknee + lthigh* np.array([-sin(q0 + q1 + qa), cos(q0 + q1 + qa)])
        psholder = pwest + lubody * np.array([-sin(q0 + q1 + q2 + qa), cos(q0 + q1 + q2 + qa)])
        phandle = pbase + lstick * np.array([-sin(qa), cos(qa)])
        if pbase[1] < 0 or psholder[1] < 0 or phandle[1] < 0:
            self.terminate()

        ret = plot_points_cmd([phandle, pbase, pknee, pwest, psholder], 0.05)
        ret = ret + plot_points_with_line_cmd([phandle, pbase, pknee, pwest, psholder])

        # sprint is shrinked
        if y <= 0:
            self.message = ["contact"]
            return ret

        d = contact_distance(x,y,qa)
        if d < 0:
            d = z0
            self.message = ["air"]
        elif d > z0:
            d = z0
            self.message = ["air"]
        else:
            self.message = ["contact"]
        pbottom = (x + d * np.sin(qa), y - d * np.cos(qa))

        ret = ret + draw_circle_cmd(pbottom, 0.05)
        ret = ret + draw_lineseg_cmd(pbottom, pbase)
        return ret

def test():
    model = Rabbit()
    dt = 0.001
    def event_handler(key, type, shifted):
        if key == 'l' and type == "DOWN":
            model.u = 0.8
        elif key == 'h' and type == "DOWN":
            model.u = -0.8
        elif type == "UP":
            model.u = 0

    run(model, event_handler, dt=dt, scale=50)

if __name__ == '__main__':
    test()
