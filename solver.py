from numpy import zeros, linspace, tanh, complex64, sin, ones, transpose, array,trapz, pi, exp
from matplotlib.pyplot import style, figure, axes, plot, text, legend, subplots, show
from celluloid import Camera
from scipy.special import softmax
import numpy as np
import scipy.integrate as spi
from numba import jit

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

def q_theor(x):
    return sin(3 * pi * x)


#Прямая задача:

    
def u_init(x):
    #return (1/2)*tanh((x-0.6)/eps)
    #return 1/(1+exp(-x/eps))
    #return softmax(x-0.6/eps)
    #return np.maximum(0, (x-0.6))
    return (1/2) * (1-tanh(-x / (2*eps)))

    

#Функция f вычисляет правую часть системы ОДУ
@jit(nopython = True)
def f(y, t, h, N, u_left, u_right, eps, q):  
    f = zeros(N-1)
    f[0] = eps*(y[1] - 2*y[0] + u_left)/h**2 + y[0]*(y[1] - u_left)/(2*h) - y[0] * q[0]
    for n in range(1,N-2):  
        f[n] = eps*(y[n+1] - 2*y[n] + y[n-1])/h**2 + y[n]*(y[n+1] - y[n-1])/(2*h) - y[n] * q[n]
    f[N-2] = eps*(u_right - 2*y[N-2] + y[N-3])/h**2 + y[N-2]*(u_right - y[N-3])/(2*h) - y[N-2] * q[N-2]
    return f  

# Функция DiagonalsPreparation подготавливает массивы, которые содержат
# элементы диагоналей трёхдиагональной матрицы
# [E - alpha*tau*f_y] 
@jit(nopython = True)
def DiagonalsPreparation(y, t, h, N, eps, tau, u_left, u_right, alpha, q):
    # Входные данные:
    # y - решение системы ОДУ в текущий момент времени
    # t - текущий момент времени
    # tau - текущий шаг по времени
    # x - сетка по пространственной координате x
    # h - шаг пространственной сетки
    # N - число интервалов пространственной сетки
    # u_left - функция, определяющая левое граничное условие
    # u_right - функция, определяющая правое граничное условие
    # alpha - параметр, определяющий схему
    # q - массив, содержащий значения функции q(x) в узлах сетки 


    a = zeros(N-1, dtype = complex64)
    b = zeros(N-1, dtype = complex64)
    c = zeros(N-1, dtype = complex64)
    a[0] = 1 - alpha * tau * (-2 * eps / h**2 + (y[1] - u_left)/(2*h) - q[0])
    c[0] = - alpha * tau * (eps / h**2 + y[0] / (2*h))
    for n in range(1, N-2) :
        b[n] = - alpha *tau* (eps / h**2 - y[n] / (2*h))
        a[n] = 1 - alpha * tau* (-2 * eps / h**2 + (y[n+1] - y[n-1]) / (2*h) -q[n])
        c[n] = - alpha * tau * (eps / h**2 + y[n] / (2*h))
    b[N-2] = - alpha * tau * (eps / h**2 - y[N-2] / (2*h))
    a[N-2] = 1 - alpha*tau*(-2 * eps / h**2 + (u_right - y[N-3]) / (2*h) -q[N-2])
    return a, b, c

@jit(nopython = True)
def TridiagonalMatrixAlgorithm(a, b, c, B):
    # Функция реализует метод прогонки (алгоритм Томаса)
    # для решения СЛАУ A X = B с трёхдиагональной матрицей

    # Входные параметры:
    # B - вектор правой части длины n
    # a, b, c - вектора длины n, содержащие элементы
    # диагоналей (b(1) и c(n) не используются)

    # [ a(1) c(1)                   ] [ X(1) ]   [ B(1) ]
    # [ b(2) a(2) c(2)              ] [ X(2) ]   [ B(2) ]
    # [      b(3) a(3) c(3)         ] [      ]   [      ]
    # [           ... ... ...       ] [ ...  ] = [ ...  ]
    # [               ... ... c(n-1)] [X(n-1)]   [B(n-1)]
    # [                   b(n) a(n) ] [ X(n) ]   [ B(n) ]
    n = len(B)
    v = zeros(n, dtype = complex64)
    X = zeros(n, dtype = complex64)
    w = a[0]
    X[0] = B[0] / w 
    for i in range(1, n) :
        v[i - 1] = c[i - 1] / w 
        w = a[i] - b[i] * v[i - 1]
        X[i] = (B[i] - b[i] * X[i - 1]) / w 
    for j in range(n-2,-1,-1):
        X[j] = X[j] - v[j] * X[j + 1]
    return X

    
# Функция PDESolving находит приближённое решение уравнения в частных производных (УрЧП/PDE)
def PDESolving(a, b, N, t_0, T, M, u_init, eps, alpha, q):
        # Входные параметры:
        # a, b - границы области по пространственно переменной x
        # N - число интервалов сетки по пространству
        # t_0, T - начальный и конечный моменты счёта
        # M - число интервалов сетки по времени
        # u_init - функция, определяющяя начальное условие
        # eps - коэффициент в исходной задаче
        # alpha - параметр, определяющий схему
        # q - массив, содержащий значения функции q(x) в узлах сетки
    h = (b - a)/N; x = linspace(a, b, N+1)
    tau = (T - t_0)/M; t = linspace(t_0, T, M+1)
    u = zeros((M + 1,N + 1))
    y = zeros(N - 1)
    u[0] = u_init(x)
    y = u_init(x[1:N])
    u_left = u_init(0)
    u_right = u_init(1)
    
    for m in range(M):
        diagonal, codiagonal_down, codiagonal_up = DiagonalsPreparation(y, t[m], h, N, eps, tau, u_left, u_right, alpha, q)
        w_1 = TridiagonalMatrixAlgorithm(diagonal, codiagonal_down, codiagonal_up, f(y, t[m] + tau/2, h, N,u_left,u_right, eps, q))

        y = y + tau*w_1.real
        u[m + 1,0] = u_left
        u[m + 1,1:N] = y
        u[m + 1,N] = u_right
    return u


#Сопряженная задача (все аналогично, оставлена без комментариев):

def psi_left(t) :
    u_left = 0
    return u_left
    
def psi_right(t) :
    u_right = 0
    return u_right

@jit(nopython = True)
def g(v, t, h, N, psi_left, psi_right, eps, q, y):  
    g = zeros(N-1) 
    g[0] = -eps * (v[1] - 2 * v[0] + psi_left) / h**2 + y[0] * (v[1] - psi_left) / (2*h) + v[0] * q[0]

    for n in range(1,N-2):  
        g[n] = -eps * (v[n+1] - 2*v[n] + v[n-1]) / h**2 + y[n] * (v[n+1] - v[n-1]) / (2*h) + v[n] * q[n]
        
    g[N-2] = -eps * (psi_right - 2*v[N-2] + v[N-3]) / h**2 + y[N-2] * (psi_right - v[N-3]) / (2*h) + v[N-2] * q[N-2]

    return g  



@jit(nopython = True)
def DiagonalsPreparation_g(y, t, h, N, eps, tau, alpha, q) :
    a = zeros(N-1, dtype = complex64)
    b = zeros(N-1, dtype = complex64)
    c = zeros(N-1, dtype = complex64)
    a[0] = 1 + alpha * tau*(2 * eps / (h**2)  + q[0])
    c[0] = + alpha * tau * (- eps / (h**2) + y[0] / (2*h))
    for n in range(1 , N-2) :
        b[n] = + alpha * tau * (- eps / h**2 - y[n] / (2*h))
        a[n] = 1 + alpha * tau * (2 * eps / h**2  + q[n])
        c[n] =  alpha * tau * (-eps / h**2 + y[n] / (2*h))
    b[N-2] =  alpha * tau * (-eps / h**2 - y[N-2] / (2*h))
    a[N-2] = 1 + alpha * tau * (2 * eps / h**2  + q[N-2])
    return a, b, c





def PDESolving_g(a, b, N, t, t_0, T, M, eps, alpha, u_xT, q, f_obs, u) :
    h = (b - a)/N
    x = linspace(a,b,N+1)
    tau = (T - t_0)/M
    psi = zeros((M + 1,N + 1))
    y = zeros(N - 1)
    psi[M, :] = -2 * (u_xT - f_obs)
    psi_left = 0
    psi_right = 0

    for m in range(M, 0, - 1):
        diagonal, codiagonal_down, codiagonal_up = DiagonalsPreparation_g(u[m, 1:N], t[m], h, N, eps, tau, alpha, q)
        w_1 = TridiagonalMatrixAlgorithm(diagonal, codiagonal_down, codiagonal_up, g(y, t[m] - tau / 2, h, N, psi_left, psi_right, eps, q, u[m, 1:N]))

        y = y - tau * w_1.real
        psi[m - 1, 0] = psi_left

        psi[m - 1, 1:N] = y
        psi[m - 1, N] = psi_right
    return psi




a = 0.; b = 1.  
t_0 = 0; T = 2
x_0 = 0.6  
eps = 10**(-1)  
alpha = (1 + 1j)/2  
N = 200; M = 50
h = (b - a)/N
tau = (T - t_0) / M 
alpha_reg = 0*0.9*10**(-6.0)


max_iter = 1000
beta = 5 #1, alpha = 10^-3     165

x = linspace(a, b, N+1)
t_arr = linspace(t_0, T, M+1)  

u = PDESolving(a, b, N, t_0, T, M, u_init, eps, alpha, q_theor(x)[1:N])
f_obs = u[M, :].copy()

J_array = zeros(max_iter)
Q = zeros((max_iter, N+1))






q = zeros(N+1)

for i in range(max_iter):
    u = PDESolving(a, b, N, t_0, T, M, u_init, eps, alpha, q[1:N])
    psi = PDESolving_g(a, b, N, t_arr, t_0, T, M, eps, alpha, u[-1], q[1:N], f_obs, u)
    gradient = trapz(u * psi, t_arr, dx = tau, axis=0) + 2 * alpha_reg * q
    q = q - beta * gradient
    Q[i] = q
    J_array[i] = spi.trapz((u[-1] - f_obs)**2, dx= tau, axis = 0) + sum(q**2) * h * alpha_reg
    print(i, J_array[i], beta)

    

q1 = q_theor(x)
style.use('dark_background')
#Анимация сходимости и убывания функционала (раскомментировать для запуска):
"""

# --- Гиф 1: сходимость q(x) ---
step = max(1, max_iter // 150)  # не больше 150 кадров

fig_anim1 = figure()
camera = Camera(fig_anim1)
ax_anim1 = axes(xlim=(a, b), ylim=(-1.1, np.max(Q)))
ax_anim1.set_xlabel('x')
ax_anim1.set_ylabel('q')

for m in range(0, max_iter):
    ax_anim1.plot(x, Q[m], color='g', marker='.', ls='-', lw=2)
    ax_anim1.plot(x, q1, color='y', ls='-', lw=2)
    ax_anim1.text(0.5, 0.95, f'Iteration: {m}',
                  transform=ax_anim1.transAxes, color='white',
                  fontsize=12, horizontalalignment='center')
    camera.snap()

print("Сохраняю convergence1.gif...")
anim1 = camera.animate(interval=15, repeat=True, blit=True)
anim1.save(os.path.join(results_dir, 'convergence1.gif'), writer='pillow', fps=10)
print("Готово!")

# --- Гиф 2: убывание функционала ---
s_list = list(range(max_iter))

fig_anim2 = figure()
camera2 = Camera(fig_anim2)
ax_anim2 = axes(xlim=(0, max_iter), ylim=(min(J_array), max(J_array)))
ax_anim2.set_xlabel('Iteration')
ax_anim2.set_ylabel('J')

for m in range(0, max_iter):
    ax_anim2.plot(s_list[:m+1], J_array[:m+1], color='r', ls='-', lw=2)
    ax_anim2.scatter(s_list[m], J_array[m], color='w', marker='o')
    ax_anim2.text(0.02, 0.95, f'Iteration: {m}',
                  transform=ax_anim2.transAxes, color='white',
                  fontsize=12, verticalalignment='bottom')
    camera2.snap()

print("Сохраняю convergence2.gif...")
anim2 = camera2.animate(interval=15, repeat=True, blit=True)
anim2.save(os.path.join(results_dir, 'convergence2.gif'), writer='pillow', fps=10)
print("Готово!")
"""
#Отрисовка графиков:
fig_q, ax_q = subplots()
ax_q.plot(x, q, color='orange', ls='-', lw=2, label='q прибл.')
ax_q.plot(x, q_theor(x), color='green', ls='--', lw=2, label='q теор.')
ax_q.set_xlabel('x')
ax_q.set_ylabel('q')
ax_q.set_title(f's = {max_iter}')
ax_q.legend()

fig_j, ax_j = subplots()
ax_j.plot(range(max_iter), J_array, color='r', ls='-', lw=2)
ax_j.set_ylim(J_array.min(), J_array.max())
ax_j.set_xlim(0, max_iter - 1)
ax_j.set_yscale('log')
ax_j.set_xlabel('Итерация')
ax_j.set_ylabel('J')
ax_j.set_title(f's = {max_iter}')


#График функционала для разных beta(раскомментировать для запуска):
"""
betas = [1, 30, 165]
colors = ['orange', 'green', 'red']

fig_beta, ax_beta = subplots()
fig_beta.patch.set_facecolor('black')
ax_beta.set_facecolor('black')

for beta_test, color in zip(betas, colors):
    q_test = zeros(N+1)
    J_test = zeros(max_iter)
    
    for i in range(max_iter):
        u_test = PDESolving(a, b, N, t_0, T, M, u_init, eps, alpha, q_test[1:N])
        psi_test = PDESolving_g(a, b, N, t_arr, t_0, T, M, eps, alpha, u_test[-1], q_test[1:N], f_obs, u_test)
        gradient_test = trapz(u_test * psi_test, t_arr, dx=tau, axis=0) + 2 * alpha_reg * q_test
        q_test = q_test - beta_test * gradient_test
        J_test[i] = spi.trapz((u_test[-1] - f_obs)**2, dx=tau, axis=0) + sum(q_test**2) * h * alpha_reg
        
    ax_beta.plot(range(max_iter), J_test, color=color, lw=2, label=f'beta = {beta_test}')
    print(f"beta = {beta_test} готово")

ax_beta.set_yscale('log')
ax_beta.set_xlabel('Итерация', color='white')
ax_beta.set_ylabel('J', color='white')
ax_beta.set_title('График функционала для различных значений beta', color='white')
ax_beta.tick_params(colors='white')
for spine in ax_beta.spines.values():
    spine.set_color('white')
ax_beta.legend(facecolor='black', labelcolor='white')
fig_beta.savefig(os.path.join(results_dir, 'beta_comparison.png'), 
                 facecolor='black', dpi=150, bbox_inches='tight')
"""

show()
