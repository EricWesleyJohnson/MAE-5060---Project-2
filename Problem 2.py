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
    E11 = 138*(10**9)   # GPa
    E22 = 8.96*(10**9)  # GPa
    V12 = 0.3           # unit-less
    G12 = 7.1*(10**9)   # GPa

    # # Independent material properties for AS/3501 graphite epoxy in US
    # E11 = 20.01 * (10**6)  # psi
    # E22 = 1.3 * (10**6)    # psi
    # V12 = 0.3              # unit-less
    # G12 = 1.03 * (10**6)   # psi

    V21 = (V12*E22)/E11      # Pg 110

    # Given mid-plane strains
    mid_plane_strains = np.array([[0.1], [0], [0]])

    # Given curvatures
    curvatures = np.array([[1], [0], [0]])

    N = 8                       # number of plies
    t_ply = [0.00015] * N       # ply thickness in m
    t_LAM = 0
    for i in range(N):
        t_LAM += t_ply[i]       # laminate thickness in m

    # Distance from laminate mid-plane to out surfaces of plies in m
    z = [0] * (N+1)
    for i in range(N+1):
        z[i] = (-t_LAM / 2) + (i * t_ply[i-1])

    # Distance from laminate mid-plane to mid-planes of plies in m
    z_mid_plane = [0] * N
    for i in range(N):
        z_mid_plane[i] = (-t_LAM / 2) - (t_ply[i]/2) + ((i+1) * t_ply[i])

    # Enter a desired ply orientation angle in degrees here:
    angle_in_degrees = [0, 0, 35, 35, 35, 35, 0, 0]

    # Ply orientation angle translated to radians to simplify equations below
    angle = [0] * N
    for i in range(N):
        angle[i] = math.radians(angle_in_degrees[i])

    # Stress Transformation (Global to Local), pg 112
    T = [0] * N
    for i in range(N):
        T[i] = np.array([[cos(angle[i])**2, sin(angle[i])**2, 2*sin(angle[i])*cos(angle[i])], [sin(angle[i])**2, cos(angle[i])**2,
         -2*sin(angle[i])*cos(angle[i])], [-sin(angle[i])*cos(angle[i]), sin(angle[i])*cos(angle[i]), cos(angle[i])**2-sin(angle[i])**2]])

    # Strain Transformation (Global-to-Local), pg 113
    T_hat = [0] * N
    for i in range(N):
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
    Q_array = lg.inv(S)  # The inverse of the S matrix
    ''' # Calculated manually, not necessary if S matrix is known, pg 110
    Q11 = E11/(1-V12*V21)
    Q12 = (V21*E11)/(1-V12*V21)
    Q21 = (V12*E22)/(1-V12*V21)
    Q22 = E22/(1-V12*V21)
    Q = np.array([[Q11, Q12, 0], [Q21, Q22, 0], [0, 0, G12]])
    '''

    # The global/laminate stiffness and compliance matrices
    Q_bar = [0] * N
    for i in range(N):
        Q_bar[i] = mm(lg.inv(T[i]), mm(Q_array,T_hat[i]))  # The global/laminate stiffness matrix, pg 114

    A = [[0]*3]*3
    for i in range(N):
        A += Q_bar[i] * t_ply[i]

    B = [[0]*3]*3
    for i in range(N):
        B += (1/2) * (Q_bar[i] * ((z[i+1]**2) - ((z[i+1] - t_ply[i])**2)))

    D = [[0] * 3] * 3
    for i in range(N):
        D += (1/3) * (Q_bar[i] * ((z[i+1] ** 3) - ((z[i+1] - t_ply[i]) ** 3)))

    ABD = np.array([[A[0][0],A[0][1],A[0][2],B[0][0],B[0][1],B[0][2]],
                    [A[1][0],A[1][1],A[1][2],B[1][0],B[1][1],B[1][2]],
                    [A[2][0],A[2][1],A[2][2],B[2][0],B[2][1],B[2][2]],
                    [B[0][0],B[0][1],B[0][2],D[0][0],D[0][1],D[0][2]],
                    [B[1][0],B[1][1],B[1][2],D[1][0],D[1][1],D[1][2]],
                    [B[2][0],B[2][1],B[2][2],D[2][0],D[2][1],D[2][2]]])

    ABD_inverse_array = lg.inv(ABD)
    stress_resultant_loads_array = mm(A, mid_plane_strains)
    stress_resultant_moments_array = mm(D, curvatures)

    # Transforming numpy array into lists for ease of formatting
    Q = Q_array.tolist()
    ABD_inverse = ABD_inverse_array.tolist()
    stress_resultant_loads = stress_resultant_loads_array.tolist()
    stress_resultant_moments = stress_resultant_moments_array.tolist()

    # Round tiny numbers to zero
    for i in range(3):
        for j in range(3):
            if 0.000000001 > A[i][j] > -0.000000001:
                A[i][j] = 0

    for i in range(3):
        for j in range(3):
            if 0.000000001 > B[i][j] > -0.000000001:
                B[i][j] = 0

    for i in range(3):
        for j in range(3):
            if 0.000000001 > D[i][j] > -0.000000001:
                D[i][j] = 0

    for i in range(6):
        for j in range(6):
            if 0.000000001 > ABD_inverse[i][j] > -0.000000001:
                ABD_inverse[i][j] = 0

    # Printing the material and its properties
    print('Material = AS/3501 graphite epoxy (in SI units)')
    print(format('E\N{SUBSCRIPT ONE}\N{SUBSCRIPT ONE} = ' + str(E11/(10**9)) + ' GPa', '<20s') + format('E\N{SUBSCRIPT TWO}\N{SUBSCRIPT TWO} = ' + str(E22/(10**9)) + ' GPa', '^20s') + format('G\N{SUBSCRIPT ONE}\N{SUBSCRIPT TWO} = ' + str(G12/(10**9)) + ' GPa', '>20s'))
    print(format('ν\N{SUBSCRIPT ONE}\N{SUBSCRIPT TWO} = ' + str(V12), '<20s') + format('ν\N{SUBSCRIPT TWO}\N{SUBSCRIPT ONE} = ' + str(round(V21,3)), '^20s'))

    # Printing the Q matrix
    print("\nThis is the local stiffness matrix [Q]:")
    print('[' + format(Q[0][0], '^12.2e') + format(Q[0][1], '^12.2e') + format(Q[0][2], '^12.2e') + ']')
    print('[' + format(Q[1][0], '^12.2e') + format(Q[1][1], '^12.2e') + format(Q[1][2], '^12.2e') + ']')
    print('[' + format(Q[2][0], '^12.2e') + format(Q[2][1], '^12.2e') + format(Q[2][2], '^12.2e') + ']')

    # Printing the Q_bar matrices
    for i in range(N):
            print("\nThis is the global stiffness matrix [Q_bar] for ply " + str(i+1) + ':')
            print('[' + format(Q_bar[i][0][0], '^12.2e') + format(Q_bar[i][0][1], '^12.2e') + format(Q_bar[i][0][2], '^12.2e') + ']')
            print('[' + format(Q_bar[i][1][0], '^12.2e') + format(Q_bar[i][1][1], '^12.2e') + format(Q_bar[i][1][2], '^12.2e') + ']')
            print('[' + format(Q_bar[i][2][0], '^12.2e') + format(Q_bar[i][2][1], '^12.2e') + format(Q_bar[i][2][2], '^12.2e') + ']')

    # Printing the A, B and D matrices
    print("\nThis is the [A] matrix:")
    print('[' + format(A[0][0], '^12.2e') + format(A[0][1], '^12.2e') + format(A[0][2], '^12.2e') + ']')
    print('[' + format(A[1][0], '^12.2e') + format(A[1][1], '^12.2e') + format(A[1][2], '^12.2e') + ']')
    print('[' + format(A[2][0], '^12.2e') + format(A[2][1], '^12.2e') + format(A[2][2], '^12.2e') + ']')
    print("\nThis is the [B] matrix:")
    print('[' + format(B[0][0], '^12.2e') + format(B[0][1], '^12.2e') + format(B[0][2], '^12.2e') + ']')
    print('[' + format(B[1][0], '^12.2e') + format(B[1][1], '^12.2e') + format(B[1][2], '^12.2e') + ']')
    print('[' + format(B[2][0], '^12.2e') + format(B[2][1], '^12.2e') + format(B[2][2], '^12.2e') + ']')
    print("\nThis is the [D] matrix:")
    print('[' + format(D[0][0], '^12.2e') + format(D[0][1], '^12.2e') + format(D[0][2], '^12.2e') + ']')
    print('[' + format(D[1][0], '^12.2e') + format(D[1][1], '^12.2e') + format(D[1][2], '^12.2e') + ']')
    print('[' + format(D[2][0], '^12.2e') + format(D[2][1], '^12.2e') + format(D[2][2], '^12.2e') + ']')

    # Printing the inverse [ABD] matrix
    print("\nThis is the [ABD]\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}")
    print('[' + format(ABD_inverse[0][0], '^12.2e') + format(ABD_inverse[0][1], '^12.2e') + format(ABD_inverse[0][3], '^12.2e') + format(ABD_inverse[0][3], '^12.2e') + format(ABD_inverse[0][4], '^12.2e') + format(ABD_inverse[0][5], '^12.2e') + ']')
    print('[' + format(ABD_inverse[1][0], '^12.2e') + format(ABD_inverse[1][1], '^12.2e') + format(ABD_inverse[1][3], '^12.2e') + format(ABD_inverse[1][3], '^12.2e') + format(ABD_inverse[1][4], '^12.2e') + format(ABD_inverse[1][5], '^12.2e') + ']')
    print('[' + format(ABD_inverse[2][0], '^12.2e') + format(ABD_inverse[2][1], '^12.2e') + format(ABD_inverse[2][3], '^12.2e') + format(ABD_inverse[2][3], '^12.2e') + format(ABD_inverse[2][4], '^12.2e') + format(ABD_inverse[2][5], '^12.2e') + ']')
    print('[' + format(ABD_inverse[3][0], '^12.2e') + format(ABD_inverse[3][1], '^12.2e') + format(ABD_inverse[3][3], '^12.2e') + format(ABD_inverse[3][3], '^12.2e') + format(ABD_inverse[3][4], '^12.2e') + format(ABD_inverse[3][5], '^12.2e') + ']')
    print('[' + format(ABD_inverse[4][0], '^12.2e') + format(ABD_inverse[4][1], '^12.2e') + format(ABD_inverse[4][3], '^12.2e') + format(ABD_inverse[4][3], '^12.2e') + format(ABD_inverse[4][4], '^12.2e') + format(ABD_inverse[4][5], '^12.2e') + ']')
    print('[' + format(ABD_inverse[5][0], '^12.2e') + format(ABD_inverse[5][1], '^12.2e') + format(ABD_inverse[5][3], '^12.2e') + format(ABD_inverse[5][3], '^12.2e') + format(ABD_inverse[5][4], '^12.2e') + format(ABD_inverse[5][5], '^12.2e') + ']')

    # Stress resultant loads
    print('\nThese are the N-components of the stress resultant:')
    print('[' + format(stress_resultant_loads[0][0], '^10.2e') + ']\n[' + format(stress_resultant_loads[1][0], '^10.2e') + ']\n[' + format(stress_resultant_loads[2][0], '^10.2e') + ']\n')

    # Stress resultant moments
    print('These are the M-components of the stress resultant:')
    print('[' + format(stress_resultant_moments[0][0], '^10.2e') + ']\n[' + format(stress_resultant_moments[1][0], '^10.2e') + ']\n[' + format(stress_resultant_moments[2][0], '^10.2e') + ']\n')

main()