import math
import numpy as np
from numpy import linalg as lg
from numpy import matmul as mm # Matrix multiplication


def sin(o):
    return math.sin(o)


def cos(o):
    return math.cos(o)


def tan(o):
    return math.tan(o)


def main():  # Plain stress approximation
    # Independent material properties for AS/3501 graphite epoxy in SI
    E11 = 138   # GPa
    E22 = 8.96  # GPa
    V12 = 0.3   # unit-less
    G12 = 7.1   # GPa

    # # Independent material properties for AS/3501 graphite epoxy in US
    # E11 = 20.01 * (10**6)   # psi
    # E22 = 1.3 * (10**6)     # psi
    # V12 = 0.3               # unit-less
    # G12 = 1.03 * (10**6)    # psi

    V21 = (V12*E22)/E11  # Pg 110

    plies = 8
    t_ply = 0.005   # inches
    t_LAM = t_ply * plies

    # Local stresses in ply 3
    local_stresses_3 = np.array([[225000],[32000],[32000]]) # in psi

    # Distance from laminate mid-plane to out surfaces of plies
    z = [0] * 9
    for i in range(9):
        z[i] = (-t_LAM / 2) + (i * t_ply)

    # Distance from laminate midplane to mid-planes of plies
    z_mid_plane = [0] * 8
    for i in range(8):
        z_mid_plane[i] = (-t_LAM / 2) - (t_ply/2) + ((i+1) * t_ply)

    # Enter a desired ply orientation angle in degrees here:
    # angle_in_degrees = [45,-45,30,-30,-30,30,-45,45]
    angle_in_degrees = [90, -45, 45, 0, 0, 45, -45, 90]

    # Ply orientation angle translated to radians to simplify equations below
    angle = [0] * 8
    for i in range(8):
        angle[i] = math.radians(angle_in_degrees[i])

    # Stress Transformation (Global to Local), pg 112
    T = [0] * 8
    for i in range(8):
        T[i] = np.array([[cos(angle[i])**2, sin(angle[i])**2, 2*sin(angle[i])*cos(angle[i])], [sin(angle[i])**2, cos(angle[i])**2,
         -2*sin(angle[i])*cos(angle[i])], [-sin(angle[i])*cos(angle[i]), sin(angle[i])*cos(angle[i]), cos(angle[i])**2-sin(angle[i])**2]])

    # Strain Transformation (Global-to-Local), pg 113
    T_hat = [0] * 8
    for i in range(8):
        T_hat[i] = np.array([[cos(angle[i])**2, sin(angle[i])**2, sin(angle[i])*cos(angle[i])], [sin(angle[i])**2, cos(angle[i])**2,
        -sin(angle[i])*cos(angle[i])], [-2*sin(angle[i])*cos(angle[i]), 2*sin(angle[i])*cos(angle[i]), cos(angle[i])**2-sin(angle[i])**2]])

    # The local/lamina compliance matrix, pg 110
    S11 = 1/E11
    S12 = -V21/E22
    S21 = -V12/E11
    S22 = 1/E22
    S33 = 1/G12
    S = np.array([[S11, S12, 0], [S21, S22, 0], [0, 0, S33]])

    # The local/lamina stiffness matrix, pg 107
    Q = lg.inv(S)  # The inverse of the S matrix
    ''' # Calculated manually, not necessary if S matrix is known, pg 110
    Q11 = E11/(1-V12*V21)
    Q12 = (V21*E11)/(1-V12*V21)
    Q21 = (V12*E22)/(1-V12*V21)
    Q22 = E22/(1-V12*V21)
    Q = np.array([[Q11, Q12, 0], [Q21, Q22, 0], [0, 0, G12]])
    '''

    # The global/laminate stiffness and complicance matrices
    Q_bar = [0] * 8
    for i in range(8):
        Q_bar[i] = mm(lg.inv(T[i]), mm(Q,T_hat[i]))  # The global/laminate stiffness matrix, pg 114

    A = [[0]*3]*3
    for k in range(8):
        A += Q_bar[k] * t_ply

    B = [[0]*3]*3
    for k in range(8):
        B += (1/2) * (Q_bar[k] * ((z[k+1]**2) - ((z[k+1] - t_ply)**2)))

    D = [[0] * 3] * 3
    for k in range(8):
        D += (1/3) * (Q_bar[k] * ((z[k+1] ** 3) - ((z[k+1] - t_ply) ** 3)))


    ABD = np.array([[A[0][0],A[0][1],A[0][2],B[0][0],B[0][1],B[0][2]], [A[1][0],A[1][1],A[1][2],B[1][0],B[1][1],B[1][2]], [A[2][0],A[2][1],A[2][2],B[2][0],B[2][1],B[2][2]],
                    [B[0][0],B[0][1],B[0][2],D[0][0],D[0][1],D[0][2]], [B[1][0],B[1][1],B[1][2],D[1][0],D[1][1],D[1][2]],[B[2][0],B[2][1],B[2][2],D[2][0],D[2][1],D[2][2]]])

    # Local Strains in ply 3
    local_strains_3 = mm(lg.inv(Q), local_stresses_3)
    print('These are the local strains in ply 3:')
    print(local_strains_3)

    # Global Strains in ply 3
    global_strains_3 = mm(lg.inv(T_hat[2]), local_strains_3)
    print('\nThese are the global strains in ply 3:')
    print(global_strains_3)

    # Laminate curvature
    curvatures = (1/z_mid_plane[2]) * global_strains_3
    print('\nThese are the curvatures')
    print(curvatures)

    # Stress resultant, specifically the M components as the Ns are 0
    M = mm(D,curvatures)
    print('\nThese are the M-components of the stress resultant:')
    print(M)

main()