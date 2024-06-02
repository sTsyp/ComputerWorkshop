"""
Декомпозиция анимации в RadarShader

Элементы анимации
1)Центр радара и его радиус
Описание: Центр радара установлен в середине экрана, а радиус определяется минимальным расстоянием до границы экрана.
Реализация: Эти параметры фиксированы и задаются при инициализации объекта.

2)Точка
Описание: Точка, движущаяся внутри радара, с определенной скоростью и направлением.
Реализация: Позиция и скорость точки хранятся в векторах. Точка инициализируется в случайном месте внутри радара, затем ее положение обновляется в зависимости от скорости и случайного ускорения.

3)Сигнальные круги
Описание: Круги, исходящие из точки и расширяющиеся со временем.
Реализация: Радиус сигнального круга увеличивается со временем, создавая эффект расширяющихся волн.

4)Вращающаяся стрелка
Описание: Стрелка, вращающаяся вокруг центра радара.
Реализация: Угол стрелки обновляется на каждом кадре в зависимости от времени. Стрелка рисуется с градиентом, меняющим интенсивность в зависимости от угла.

5)Горизонтальные и вертикальные треугольники
Описание: Два набора треугольников, движущихся горизонтально и вертикально.
Реализация: Позиции треугольников обновляются на каждом кадре в зависимости от синусоидальных функций, создавая плавное движение.

6)Фон и линии
Описание: Фон радара с концентрическими кругами и крестовинами.
Реализация: Отрисовка концентрических кругов и линий в центре экрана для создания эффекта радара.

Взаимосвязи элементов
1)Центр радара и его радиус
Связи: Определяют область, в которой могут двигаться белая точка, треугольники и вращаться стрелка.
Реализация: Эти параметры используются для вычисления позиций и ограничений движения.

2)Точка
Связи: Влияет на сигнальные круги, которые исходят из её текущей позиции.
Реализация: Позиция  точки обновляется в каждом кадре, её скорость ограничивается для плавности движения.

3)Сигнальные круги
Связи: Расширяются из позиции  точки, создавая эффект волн.
Реализация: Радиус круга увеличивается с течением времени, центр круга совпадает с текущей позицией точки.

4)Вращающаяся стрелка
Связи: Вращается вокруг центра радара и визуализируется в зависимости от угла и радиуса.
Реализация: Угол стрелки увеличивается с течением времени, цвет меняется с градиентом.

5)Горизонтальные и вертикальные треугольники
Связи: Двигаются синхронно, изменяя свою позицию по синусоидальной траектории.
Реализация: Позиции треугольников обновляются в зависимости от времени, создавая плавное движение по синусоиде.

6)Фон и линии
Связи: Обеспечивают статический фон для радара, на котором отображаются все динамические элементы.
Реализация: Отрисовка концентрических кругов и линий осуществляется в начале каждой итерации для создания статичного фона.
"""
import taichi as ti
from base_shader import BaseShader
import math

@ti.data_oriented
class RadarShader(BaseShader):
    def __init__(self, res):
        """
        Этот класс представляет собой радарный шейдер, который создает визуализацию в виде радара с движущимися элементами.
        Аргументы:
            res (tuple): Кортеж, представляющий разрешение шейдера (ширина, высота).
        Атрибуты:
            time (ti.field): Поле, представляющее время.
            center (tuple): Кортеж, представляющий координаты центра радара.
            radius (float): Радиус радара.
            white_point_pos (ti.Vector.field): Поле, представляющее позицию белой точки.
            white_point_velocity (ti.Vector.field): Поле, представляющее скорость белой точки.
            white_point_radius (float): Радиус белой точки.
            max_signal_radius (float): Максимальный радиус сигнального круга.
            arrow_radius (float): Радиус стрелки.
            horizontal_positions (ti.field): Поле, представляющее горизонтальные позиции.
            horizontal_positions1 (ti.field): Поле, представляющее горизонтальные позиции для другого  треугольника.
            vertical_positions (ti.field): Поле, представляющее вертикальные позиции.
            vertical_positions1 (ti.field): Поле, представляющее вертикальные позиции для другого  треугольника.
            horizontal_range (tuple): Кортеж, представляющий диапазон горизонтального движения.
            horizontal_range1 (tuple): Кортеж, представляющий диапазон горизонтального движения для другого  треугольника.
            vertical_range (tuple): Кортеж, представляющий диапазон вертикального движения.
            vertical_range1 (tuple): Кортеж, представляющий диапазон вертикального движения для другого  треугольника.
        """
        super().__init__("Radar Shader", res)
        self.time = ti.field(dtype=ti.f32, shape=())
        self.center = (res[0] // 2, res[1] // 2)
        self.radius = min(self.center)
        self.white_point_pos = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.white_point_velocity = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.white_point_radius = 5  # Радиус точки
        self.max_signal_radius = 20  # Максимальный радиус сигнального круга
        self.arrow_radius = self.radius * 0.75  # Радиус стрелки
        # Позиции треугольников
        self.horizontal_positions = ti.field(dtype=ti.f32, shape=2)
        self.horizontal_positions1 = ti.field(dtype=ti.f32, shape=2)
        self.vertical_positions = ti.field(dtype=ti.f32, shape=2)
        self.vertical_positions1 = ti.field(dtype=ti.f32, shape=2)
        # Диапазон движения треугольников
        self.horizontal_range = (self.radius * 1, self.radius * 0.9)
        self.horizontal_range1 = (self.radius*1, self.radius * 0.9)
        self.vertical_range = (self.radius * 1, self.radius * 0.9)
        self.vertical_range1 = (self.radius*1, self.radius * 0.9)
        # Инициализация скорости белой точки
        self.white_point_velocity[None] = [0.0, 0.0]

    @ti.kernel
    def initialize_point(self):
        """
        Инициализирует позицию точки в пределах радара.
        Аргументы:
            Нет
        Возвращает:
            Нет
        """
        random_angle = ti.random(ti.f32) * 2 * math.pi
        random_dist = ti.random(ti.f32) * self.radius
        self.white_point_pos[None] = [
            self.center[0] + random_dist * ti.cos(random_angle),
            self.center[1] + random_dist * ti.sin(random_angle)
        ]

    @ti.func
    def point_in_triangle(self, p, p0, p1, p2):
        """
        Определяет, находится ли точка внутри треугольника.
        Аргументы:
            p (tuple): Координаты точки.
            p0 (ti.Vector): Координаты первой вершины треугольника.
            p1 (ti.Vector): Координаты второй вершины треугольника.
            p2 (ti.Vector): Координаты третьей вершины треугольника.
        Возвращает:
            bool: True, если точка находится внутри треугольника, False в противном случае.
        """
        dX = p[0] - p2[0]
        dY = p[1] - p2[1]
        dX21 = p2[0] - p1[0]
        dY12 = p1[1] - p2[1]
        D = dY12 * (p0[0] - p2[0]) + dX21 * (p0[1] - p2[1])
        s = dY12 * dX + dX21 * dY
        t = (p2[1] - p0[1]) * dX + (p0[0] - p2[0]) * dY
        inside = False
        if D < 0:
            inside = s <= 0 and t <= 0 and s + t >= D
        else:
            inside = s >= 0 and t >= 0 and s + t <= D
        return inside

    @ti.kernel
    def render(self, out: ti.template()):
        """
        Визуализирует радар.
        Аргументы:
            out (ti.template): Выходное поле для визуализации.
        Возвращает:
            Нет
        """
        # Очистка экрана
        for i, j in ti.ndrange(*self.res):
            out[i, j] = ti.Vector([0.0, 0.0, 0.0])

        # Отрисовка концентрических кругов и диаметров
        step = 100  
        for i, j in ti.ndrange(*self.res):
            dist = ((i - self.center[0]) ** 2 + (j - self.center[1]) ** 2) ** 0.5
            if int(dist) % step < 1:
                out[i, j] = ti.Vector([0.3, 0.3, 0.3])  
            if i == self.center[0] or j == self.center[1]:
                out[i, j] = ti.Vector([0.3, 0.3, 0.3])  

        # Отрисовка точки
        white_point_i = int(self.white_point_pos[None][0])
        white_point_j = int(self.white_point_pos[None][1])
        for x, y in ti.ndrange((-self.white_point_radius, self.white_point_radius + 1),
                               (-self.white_point_radius, self.white_point_radius + 1)):
            if x ** 2 + y ** 2 <= self.white_point_radius ** 2:
                px = white_point_i + x
                py = white_point_j + y
                if 0 <= px < self.res[0] and 0 <= py < self.res[1]:
                    out[px, py] = ti.Vector([1.0, 0.0, 0.0])

        # Отрисовка сигнальных кругов от точки
        signal_radius = (self.time[None] * 50) % self.max_signal_radius
        for i, j in ti.ndrange(*self.res):
            dist = ((i - self.white_point_pos[None][0]) ** 2 + (j - self.white_point_pos[None][1]) ** 2) ** 0.5
            if abs(dist - signal_radius) < 2:
                out[i, j] = ti.Vector([1.0, 0.0, 0.0]) 

        tri_base = 15
        tri_height = 10
        
        # Отрисовка горизонтальных треугольников
        for k in ti.ndrange(1):
            pos = self.horizontal_positions[k]
            p11 = ti.Vector([pos + tri_height, self.center[1]])
            p21 = ti.Vector([pos, self.center[1] + tri_base / 2])  
            p31 = ti.Vector([pos, self.center[1] - tri_base / 2])

            for x, y in ti.ndrange(*self.res):
                if self.point_in_triangle((x, y), p11, p21, p31):
                    out[x, y] = ti.Vector([1.0, 1.0, 1.0]) 

        for k in ti.ndrange(1):
            pos1 = self.horizontal_positions1[k]
            p12 = ti.Vector([pos1, self.center[1]])
            p22 = ti.Vector([pos1 + tri_height, self.center[1] - tri_base / 2])
            p32 = ti.Vector([pos1 + tri_height, self.center[1] + tri_base / 2])
            for x, y in ti.ndrange(*self.res):
                if self.point_in_triangle((x, y), p12, p22, p32):
                    out[x, y] = ti.Vector([1.0, 1.0, 1.0]) 
        
        # Отрисовка вертикальных треугольников
        for k in ti.ndrange(1):
            pos2 = self.vertical_positions[k]
            p13 = ti.Vector([self.center[0], pos2+ tri_height])
            p23 = ti.Vector([self.center[0] + tri_base / 2, pos2 ])
            p33 = ti.Vector([self.center[0] - tri_base / 2, pos2 ])

            for x, y in ti.ndrange(*self.res):
                if self.point_in_triangle((x, y), p13, p23, p33):
                    out[x, y] = ti.Vector([1.0, 1.0, 1.0]) 
        for k in ti.ndrange(1):
            pos3 = self.vertical_positions1[k]
            p14 = ti.Vector([self.center[0], pos3 ])
            p24 = ti.Vector([self.center[0] - tri_base / 2, pos3 + tri_height])
            p34 = ti.Vector([self.center[0] + tri_base / 2, pos3 + tri_height])

            for x, y in ti.ndrange(*self.res):
                if self.point_in_triangle((x, y), p14, p24, p34):
                    out[x, y] = ti.Vector([1.0, 1.0, 1.0])  

        # Отрисовка вращающейся стрелки с градиентом
        current_angle = (self.time[None] * 2 * math.pi) % (2 * math.pi)
        for i, j in ti.ndrange(*self.res):
            dist = ((i - self.center[0]) ** 2 + (j - self.center[1]) ** 2) ** 0.5
            if dist < self.arrow_radius:  
                angle = ti.atan2(j - self.center[1], i - self.center[0])
                if angle < 0:
                    angle += 2 * math.pi
                angle_diff = abs(angle - current_angle)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                if angle_diff < 0.1:
                    gradient = 1 - (angle_diff / 0.1)
                    out[i, j] = ti.Vector([0.0, 0.8 * gradient, 1.0 * gradient])

    @ti.kernel
    def update_point(self):
        """
        Обновляет позицию точки с плавным движением.
        Аргументы:
            Нет
        Возвращает:
            Нет
        """
        # Обновление позиции  точки с плавным перемещением
        acceleration = ti.Vector([ti.random(ti.f32) * 2-3, ti.random(ti.f32) * 2-3])
        self.white_point_velocity[None] += acceleration * 0.1

        # Ограничение максимальной скорости
        speed_limit = 0.2
        speed = self.white_point_velocity[None].norm()
        if speed > speed_limit:
            self.white_point_velocity[None] = self.white_point_velocity[None].normalized() * speed_limit

        # Обновление позиции
        new_pos = self.white_point_pos[None] + self.white_point_velocity[None]
        dist_from_center = ((new_pos[0] - self.center[0]) ** 2 + (new_pos[1] - self.center[1]) ** 2) ** 0.5
        
        # Убедимся, что точка остается внутри радара
        #if dist_from_center < self.radius:
        self.white_point_pos[None] = new_pos
        #else:
            # Отражение от границы радара
        #    self.white_point_pos[None] = -new_pos
    
    @ti.kernel
    def update_positions(self):
        """
        Обновляет позиции треугольников внутри радара.
        Аргументы:
            Нет
        Возвращает:
            Нет
        """
        t = self.time[None]
        horizontal_amplitude = (self.horizontal_range[0] - self.horizontal_range[1]) 
        horizontal_center = 40
        self.horizontal_positions[0] = horizontal_center + horizontal_amplitude * ti.sin(t * math.pi)

        horizontal_amplitude1 = (self.horizontal_range1[0] - self.horizontal_range1[1])
        horizontal_center1 = 750
        self.horizontal_positions1[0] = horizontal_center1 + horizontal_amplitude1 * ti.sin(t * math.pi + math.pi)

        vertical_amplitude = (self.vertical_range[0] - self.vertical_range[1])
        vertical_center = 40
        self.vertical_positions[0] = vertical_center + vertical_amplitude * ti.sin(t * math.pi)

        vertical_amplitude1 = (self.vertical_range1[0] - self.vertical_range1[1])
        vertical_center1 = 750
        self.vertical_positions1[0] = vertical_center1 + vertical_amplitude1 * ti.sin(t * math.pi + math.pi)
    
    def main_image(self, out: ti.template()):
        """
        Обновляет время и визуализирует основное изображение радара.
        Аргументы:
            out (ti.template): Выходное поле для визуализации основного изображения.
        Возвращает:
            Нет
        """
        self.time[None] += 0.004
        self.update_point()
        self.update_positions()
        self.render(out)

res = (800, 800)  
shader = RadarShader(res)
shader.initialize_point()  # Инициализация случайной позиции точки
out = ti.Vector.field(3, dtype=ti.f32, shape=res)

gui = ti.GUI("Radar Shader", res)
while gui.running:
    shader.main_image(out)
    gui.set_image(out)
    gui.show()
