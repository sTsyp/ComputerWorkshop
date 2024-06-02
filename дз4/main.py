import numpy as np
from vispy import app, scene
from vispy.geometry import Rect
import imageio as imageio
import config as cfg
from funcs import init_boids, directions, propagate, flocking, periodic_walls, wall_avoidance, simulation_step
from vispy.scene.visuals import Text
app.use_app('pyglet')

writer = imageio.get_writer('video2.mp4', fps=60)


c_names = cfg.coeffs
c = np.array(list(c_names))

boids = np.zeros((cfg.N, 6), dtype=np.float64)
init_boids(boids, cfg.asp, vrange=cfg.vrange)

canvas = scene.SceneCanvas(show=True, size=(cfg.w, cfg.h))
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, cfg.asp, 1))

color_dict = {"red": (1, 0, 0, 1), "green": (0, 1, 0, 1), "blue": (0, 0, 1, 1), "black": (0, 0, 0, 1)}

color_arr = [color_dict["blue"]] * cfg.N

color_arr[0] = color_dict["red"]
canvas.bgcolor = (1, 1, 1, 1)
arrows = scene.Arrow(arrows=directions(boids, cfg.dt),
                     arrow_color=color_arr,
                     arrow_size=10,
                     connect='segments',
                     parent=view.scene)

canvas.title = "simulation"
txt = Text(parent=canvas.scene, color='green', face='Consolas', bold=True)
txt.pos = canvas.size[0] // 12, canvas.size[1] // 14
txt.font_size = 12
txt_const = Text(parent=canvas.scene, color='green', face='Consolas', bold=True)
txt_const.pos = canvas.size[0] // 12, canvas.size[1] // 10
txt_const.font_size = 10
general_info = f"boids: {cfg.N}\n"
general_info += f"coefficients: {cfg.coeffs}\n"
txt_const.text = general_info



def update(event):
    """
    Обновляет состояние симуляции на каждом кадре.
    Args:
        event: Событие таймера (не используется).
    Returns:
        None
    """
    calculated_data = flocking(boids, cfg.perception, cfg.coeffs, cfg.asp, cfg.vrange, cfg.better_walls_w, cnt_rely_on=cfg.cnt_rely_on)
    propagate(boids, cfg.dt, cfg.vrange, cfg.arange)
    periodic_walls(boids, cfg.asp)
    wall_avoidance(boids, cfg.asp)
    print(calculated_data["mask_see"].shape)
    color_arr = np.array([color_dict["black"]] * cfg.N)

    color_arr[calculated_data["mask_see"][0]] = color_dict["green"]
    color_arr[calculated_data["mask_rely"][0]] = color_dict["blue"]

    color_arr[0] = color_dict["red"]

    arrows.arrow_color = color_arr
    arrows.set_data(arrows=directions(boids, cfg.dt))
    txt.text = f'FPS: {canvas.fps:.2f}'

    if cfg.frame < 2700:

        cadr = canvas.render(alpha=False)
        writer.append_data(cadr)
    else:
        writer.close()
        app.quit()

    cfg.frame += 1


if __name__ == '__main__':
    timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()