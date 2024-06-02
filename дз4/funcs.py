import numpy as np
from numba import njit, prange
from typing import Tuple
from typing import Union
import config as cfg
def init_boids(boids: np.ndarray, asp: float, vrange: Tuple[float, float]):
    """
    Инициализирует положение и скорость боидов случайным образом в заданных диапазонах.
    Args:
        boids (np.ndarray): Массив, содержащий положение и скорость боидов.
        asp (float): Соотношение сторон области моделирования.
        vrange (Tuple[float, float]): Диапазон скоростей для боидов.
    Returns:
        None
    """
    n = boids.shape[0]
    rng = np.random.default_rng()
    boids[:, 0] = rng.uniform(0., asp, size=n)
    boids[:, 1] = rng.uniform(0., 1., size=n)
    alpha = rng.uniform(0, 2 * np.pi, size=n)
    v = rng.uniform(*vrange, size=n)
    c, s = np.cos(alpha), np.sin(alpha)
    boids[:, 2] = v * c
    boids[:, 3] = v * s


def directions(boids: np.ndarray, dt: float) -> np.ndarray:
    """
    Вычисляет будущее положение боидов на основе их текущего положения и скорости.
    Args:
        boids (np.ndarray): Массив, содержащий положение и скорость боидов.
        dt (float): Шаг времени для моделирования.
    Returns:
        np.ndarray: Массив, содержащий будущее положение боидов.
    """
    return np.hstack((
        boids[:, :2] - dt * boids[:, 2:4],
        boids[:, :2]
    ))


def vclip(v: np.ndarray, vrange: Tuple[float, float]):
    """
    Обрезает скорости боидов, чтобы они оставались в заданных диапазонах.
    Args:
        v (np.ndarray): Массив, содержащий скорости боидов.
        vrange (Tuple[float, float]): Диапазон скоростей для боидов.
    Returns:
        None
    """
    norm = np.linalg.norm(v, axis=1)
    mask = norm > vrange[1]
    if np.any(mask):
        v[mask] *= (vrange[1] / norm[mask]).reshape(-1, 1)


def propagate(boids: np.ndarray,
              dt: float,
              vrange: Tuple[float, float],
              arange: Tuple[float, float]):
    """
    Обновляет положение и скорость боидов в соответствии с их текущими значениями и ускорением.
    Args:
        boids (np.ndarray): Массив, содержащий положение и скорость боидов.
        dt (float): Шаг времени для обновления.
        vrange (Tuple[float, float]): Диапазон скоростей для боидов.
        arange (Tuple[float, float]): Диапазон ускорений для боидов.
    Returns:
        None
    """
    vclip(boids[:, 4:6], arange)
    boids[:, 2:4] += dt * boids[:, 4:6]
    vclip(boids[:, 2:4], vrange)
    boids[:, 0:2] += dt * boids[:, 2:4]


@njit()
def distances(vecs: np.ndarray) -> np.ndarray:
    """
    Вычисляет расстояния между всеми парами векторов.
    Args:
        vecs (np.ndarray): Массив векторов.
    Returns:
        np.ndarray: Матрица расстояний между парами векторов.
    """
    n, m = vecs.shape
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(m):
                s += (vecs[i][k] - vecs[j][k]) ** 2
            D[i, j] = np.sqrt(s)
    return D


@njit()
def cohesion(boids: np.ndarray,
             idx: int,
             neigh_mask: np.ndarray,
             perception: float) -> np.ndarray:
    """
    Вычисляет вектор согласования для заданного боида.
    Args:
        boids (np.ndarray): Массив, содержащий положение и скорость боидов.
        idx (int): Индекс текущего боида.
        neigh_mask (np.ndarray): Маска, определяющая соседей боида.
        perception (float): Радиус восприятия боида.
    Returns:
        np.ndarray: Вектор согласования для боида.
    """
    center = np.mean(boids[neigh_mask, :2])
    a = (center - boids[idx, :2]) / perception
    return a


@njit()
def norm(vec: np.ndarray) -> np.ndarray:
    """
    Вычисляет норму вектора.
    Args:
        vec (np.ndarray): Вектор.
    Returns:
        np.ndarray: Норма вектора.
    """
    return sum([c ** 2 for c in vec]) ** 0.5


@njit()
def separation(boids: np.ndarray,
               idx: int,
               neigh_mask: np.ndarray,
               perception: float) -> np.ndarray:
    """
    Вычисляет вектор разделения для заданного боида.
    Args:
        boids (np.ndarray): Массив, содержащий положение и скорость боидов.
        idx (int): Индекс текущего боида.
        neigh_mask (np.ndarray): Маска, определяющая соседей боида.
        perception (float): Радиус восприятия боида.
    Returns:
        np.ndarray: Вектор разделения для боида.
    """
    dx = np.mean(boids[neigh_mask, 0] - boids[idx, 0])
    dy = np.mean(boids[neigh_mask, 1] - boids[idx, 1])

    length_squared = dx ** 2 + dy ** 2 + 1  # Add 1 to avoid division by zero
    return -np.array([dx, dy]) / length_squared


@njit()
def alignment(boids: np.ndarray,
              idx: int,
              neigh_mask: np.ndarray,
              vrange: tuple) -> np.ndarray:
    """
    Вычисляет вектор выравнивания для заданного боида.
    Args:
        boids (np.ndarray): Массив, содержащий положение и скорость боидов.
        idx (int): Индекс текущего боида.
        neigh_mask (np.ndarray): Маска, определяющая соседей боида.
        vrange (tuple): Диапазон скоростей боидов.
    Returns:
        np.ndarray: Вектор выравнивания для боида.
    """
    v_mean = np.mean(boids[neigh_mask, 2:4])
    a = (v_mean - boids[idx, 2:4]) / (2 * vrange[1])
    return a


@njit()
def walls(boids: np.ndarray, asp: float, param: int):
    """
    Вычисляет вектора отталкивания от стен для всех боидов.
    Args:
        boids (np.ndarray): Массив, содержащий положение и скорость боидов.
        asp (float): Соотношение сторон области моделирования.
        param (int): Параметр для вычисления отталкивания от стен.
    Returns:
        np.ndarray: Массив векторов отталкивания от стен для всех боидов.
    """
    c = 1
    x = boids[:, 0]
    y = boids[:, 1]
    order = param

    a_left = 1 / (np.abs(x) + c) ** order
    a_right = -1 / (np.abs(x - asp) + c) ** order

    a_bottom = 1 / (np.abs(y) + c) ** order
    a_top = -1 / (np.abs(y - 1.) + c) ** order

    return np.column_stack((a_left + a_right, a_bottom + a_top))


@njit()
def smoothstep(edge0: float, edge1: float, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Вычисляет сглаженное значение по функции Smoothstep.
    Args:
        edge0 (float): Начальное значение диапазона.
        edge1 (float): Конечное значение диапазона.
        x (Union[np.ndarray, float]): Значение входа.
    Returns:
        Union[np.ndarray, float]: Сглаженное значение.
    """
    x = np.clip((x - edge0) / (edge1 - edge0), 0., 1.)
    return x * x * (3.0 - 2.0 * x)


@njit()
def better_walls(boids: np.ndarray, asp: float, param: float):
    """
    Вычисляет вектора отталкивания от стен для всех боидов с использованием сглаживающей функции.
    Args:
        boids (np.ndarray): Массив, содержащий положение и скорость боидов.
        asp (float): Соотношение сторон области моделирования.
        param (float): Параметр сглаживания.
    Returns:
        np.ndarray: Массив векторов отталкивания от стен для всех боидов.
    """
    x = boids[:, 0]
    y = boids[:, 1]
    w = param

    a_left = smoothstep(asp * w, 0.0, x)
    a_right = -smoothstep(asp * (1.0 - w), asp, x)

    a_bottom = smoothstep(w, 0.0, y)
    a_top = -smoothstep(1.0 - w, 1.0, y)

    return np.column_stack((a_left + a_right, a_bottom + a_top))


@njit(parallel=True)
def flocking(boids: np.ndarray,
             perception: float,
             coeffs: np.ndarray,
             asp: float,
             vrange: tuple,
             order: int,
             cnt_rely_on: int):
    """
    Реализует поведение стай боидов.
    Args:
        boids (np.ndarray): Массив, содержащий положение и скорость боидов.
        perception (float): Радиус восприятия боида.
        coeffs (np.ndarray): Коэффициенты для вычисления направления движения.
        asp (float): Соотношение сторон области моделирования.
        vrange (tuple): Диапазон скоростей боидов.
        order (int): Параметр для управления поведением стен.
        cnt_rely_on (int): Количество боидов, от которых боид может получить информацию.
    Returns:
        dict: Словарь, содержащий маски ближайших боидов для взаимодействия.
    """
    N = boids.shape[0]
    DistMatrix = np.zeros((N, N))

    DistMatrix = distances(boids[:, 0:2])

    for i in prange(N):
        DistMatrix[i, i] = perception + 1

    mask = DistMatrix < perception
    max_cnt = cnt_rely_on
    for i in range(N):
        sorted_indices = np.argsort(DistMatrix[i])
        DistMatrix[i, sorted_indices[max_cnt:]] = 100


    mask_rely = DistMatrix < perception

    wal = better_walls(boids, asp, order)
    for i in prange(N):
        if not np.any(mask[i]):
            coh = np.zeros(2)
            alg = np.zeros(2)
            sep = np.zeros(2)
        else:
            coh = cohesion(boids, i, mask[i], perception)
            alg = alignment(boids, i, mask[i], vrange)
            sep = separation(boids, i, mask[i], perception)
        a = coeffs[0] * coh + coeffs[1] * alg + \
            coeffs[2] * sep + coeffs[3] * wal[i]
        boids[i, 4:6] = a
    return {
        "mask_rely": mask_rely,
        "mask_see": mask
    }


def periodic_walls(boids: np.ndarray, asp: float):
    """
    Обрабатывает положение боидов относительно периодических стен.
    Args:
        boids (np.ndarray): Массив, содержащий положение и скорость боидов.
        asp (float): Соотношение сторон области моделирования.
    Returns:
        None
    """
    boids[:, 0:2] %= np.array([asp, 1.])


def wall_avoidance(boids: np.ndarray, asp: float):
    """
    Реализует избегание стен в логике ускорения для боидов.
    Args:
        boids (np.ndarray): Массив, содержащий положение и скорость боидов.
        asp (float): Соотношение сторон области моделирования.
    Returns:
        None
    """
    left = np.abs(boids[:, 0])
    right = np.abs(asp - boids[:, 0])
    bottom = np.abs(boids[:, 1])
    top = np.abs(1 - boids[:, 1])
    ax = 1. / left ** 2 - 1. / right ** 2
    ay = 1. / bottom ** 2 - 1. / top ** 2
    boids[:, 4:6] += np.column_stack((ax, ay))

def simulation_step(boids: np.ndarray,
                    asp: float,
                    perception: float,
                    coeffs: np.ndarray,
                    v_range: tuple,
                    dt: float) -> None:
    """
    Выполняет один шаг симуляции для стай боидов.
    Args:
        boids (np.ndarray): Массив, содержащий положение и скорость боидов.
        asp (float): Соотношение сторон области моделирования.
        perception (float): Радиус восприятия боида.
        coeffs (np.ndarray): Коэффициенты для вычисления направления движения.
        v_range (tuple): Диапазон скоростей боидов.
        dt (float): Шаг времени для моделирования.
    Returns:
        None
    """
    flocking(boids, cfg.perception, cfg.coeffs, cfg.asp, cfg.vrange, cfg.better_walls_w, cnt_rely_on=cfg.cnt_rely_on)
    propagate(boids, cfg.dt, cfg.vrange, cfg.arange)
    periodic_walls(boids, cfg.asp)
    wall_avoidance(boids, cfg.asp)