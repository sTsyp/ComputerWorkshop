import taichi as ti
import time
from typing import Optional, Tuple
import math

ti.init(arch=ti.x64)  # Инициализация Taichi

@ti.data_oriented
class BaseShader:
    def __init__(self, title: str, res: Optional[Tuple[int, int]] = None, gamma: float = 2.2):
        """
        Этот класс представляет собой базовый шейдер для создания различных визуализаций с помощью Taichi.
        Аргументы:
            title (str): Название шейдера.
            res (Optional[Tuple[int, int]]): Разрешение шейдера (по умолчанию (1280, 720)).
            gamma (float): Гамма-коррекция (по умолчанию 2.2).
        Атрибуты:
            title (str): Название шейдера.
            res (Tuple[int, int]): Разрешение шейдера.
            resf (ti.Vector): Вектор, представляющий разрешение в формате ti.Vector.
            pixels (ti.Vector.field): Поле для хранения пикселей изображения.
            gamma (float): Гамма-коррекция.
        """
        self.title = title
        self.res = res if res is not None else (1280, 720)
        self.resf = ti.Vector([self.res[0], self.res[1]])
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=self.res)
        self.gamma = gamma

    @ti.kernel
    def render(self, t: ti.f32):
        """
        Выполняет рендеринг изображения.
        Аргументы:
            t (ti.f32): Время с начала выполнения программы.
        Возвращает:
            Нет
        """
        for fragCoord in ti.grouped(self.pixels):
            uv = (fragCoord - 0.5 * self.resf) / self.resf.y
            col = self.main_image(uv, t)
            if self.gamma > 0.0:
                col = col ** (1 / self.gamma)
            self.pixels[fragCoord] = col

    @ti.func
    def main_image(self, uv, t):
        """
        Вычисляет цвет пикселя на основе координат и времени.
        Аргументы:
            uv (ti.Vector): Координаты пикселя.
            t (ti.f32): Время с начала выполнения программы.
        Возвращает:
            ti.Vector: Цвет пикселя в формате ti.Vector.
        """
        col = ti.Vector([0.0, 0.0, 0.0])
        col[0], col[1] = uv + 0.5
        return col

    def main_loop(self):
        """
        Запускает главный цикл программы, который обновляет изображение и обрабатывает события GUI.
        Аргументы:
            Нет
        Возвращает:
            Нет
        """
        gui = ti.GUI(self.title, res=self.res, fast_gui=True)
        start = time.time()

        while gui.running:
            if gui.get_event(ti.GUI.PRESS):
                if gui.event.key == ti.GUI.ESCAPE:
                    break

            t = time.time() - start
            self.render(t)
            gui.set_image(self.pixels)
            gui.show()

        gui.close()

if __name__ == "__main__":
    shader = BaseShader("Base shader")
    shader.main_loop()
