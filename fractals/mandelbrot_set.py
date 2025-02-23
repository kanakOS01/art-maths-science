"""
https://www.youtube.com/watch?v=FFftmWSzgmk
https://en.wikipedia.org/wiki/Mandelbrot_set
https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set
"""

import taichi as ti
from taichi.math import cmul, dot, log2, vec2, vec3
from numpy import linspace, array
from scipy.interpolate import pchip_interpolate
from PIL import Image
ti.init(arch=ti.gpu)


def gen_static_mandelbrot_set():
    w, h = 800, 640
    pixels = ti.Vector.field(3, float, shape=(w, h))

    @ti.func
    def set_color(z, i):
        v = log2(i + 1 - log2(log2(z.norm()))) / 5
        col = vec3(0)
        if v < 1:
            col = vec3(v**4, v**2.5, v)
        else:
            v = ti.max(0., 2 - v)
            col = vec3(v, v**1.5, v**3)
        return col

    @ti.kernel
    def render():
        for i, j in pixels:
            c = 2 * vec2(i, j) / h - vec2(1.8, 1)
            z = vec2(0)
            count = 0
            while count < 100 and dot(z, z) < 50:
                z = cmul(z, z) + c
                count += 1

            if count == 100:
                pixels[i, j] = [0, 0, 0]
            else:
                pixels[i, j] = set_color(z, count)

    render()

    img = Image.fromarray((255 * pixels.to_numpy()).astype('uint8')).transpose(Image.Transpose.TRANSPOSE)
    img.show()    


def mandelbrot_viewer():
    WIDTH, HEIGHT = 1200, 600
    ZOOM_RATE = 1.2
    MAX_ITER = 1000
    COLORMAP_SIZE = 1000
    COLORMAP = pchip_interpolate(
        [0, 0.16, 0.42, 0.6425, 0.8575, 1],
        array([[0, 7, 100], [32, 107, 203], [237, 255, 255], [255, 170, 0], [0, 2, 0], [0, 7, 100]]) / 255,
        linspace(0, 1, COLORMAP_SIZE)
    ).flatten()
    
    zoom = 250
    center_x = -0.5
    center_y = 0

    pixels = ti.Vector.field(3, dtype=float, shape=(WIDTH, HEIGHT))
    gui = ti.GUI("Mandelbrot Viewer", res=(WIDTH, HEIGHT))

    @ti.func
    def iter(x, y):
        c = ti.Vector([x, y])
        z = c
        count = 0.
        
        while count < MAX_ITER and z.norm() <= 2:
            z = ti.Vector([z[0]**2 - z[1]**2, z[0] * z[1] * 2]) + c
            count += 1.0
        
        if count < MAX_ITER:
            count += 1.0 - ti.log(ti.log(ti.cast(z.norm(), ti.f32)) / ti.log(2)) / ti.log(2)
        return count


    @ti.kernel
    def paint(center_x: ti.f64, center_y: ti.f64, zoom: ti.f64, colormap: ti.types.ndarray()):
        for i, j in pixels:
            x = center_x + (i - WIDTH / 2 + 0.5) / zoom
            y = center_y + (j - HEIGHT / 2 + 0.5) / zoom
            index = int(iter(x, y) / MAX_ITER * COLORMAP_SIZE)
            for k in ti.static(range(3)):
                pixels[i, j][k] = colormap[3 * index + k]

    # GUI
    # gui.fps_limit = 10
    while gui.running:
        for e in gui.get_events(gui.PRESS, gui.MOTION):
            if e.key == ti.GUI.LMB:
                # left click: record position
                mouse_x0, mouse_y0 = gui.get_cursor_pos()
                center_x0, center_y0 = center_x, center_y
            elif e.key == ti.GUI.WHEEL:
                # scroll: zoom
                mouse_x, mouse_y = gui.get_cursor_pos()
                if e.delta[1] > 0:
                    zoom_new = zoom * ZOOM_RATE
                elif e.delta[1] < 0:
                    zoom_new = zoom / ZOOM_RATE
                center_x += (mouse_x - 0.5) * WIDTH * (1 / zoom - 1 / zoom_new)
                center_y += (mouse_y - 0.5) * HEIGHT * (1 / zoom - 1 / zoom_new)
                zoom = zoom_new
            elif e.key == ti.GUI.SPACE:
                # space: print info
                print(f'center_x={center_x}, center_y={center_y}, zoom={zoom}')
        if gui.is_pressed(ti.GUI.LMB):
            # drag: move
            mouse_x, mouse_y = gui.get_cursor_pos()
            center_x = center_x0 + (mouse_x0 - mouse_x) * WIDTH / zoom
            center_y = center_y0 + (mouse_y0 - mouse_y) * HEIGHT / zoom

        paint(center_x, center_y, zoom, COLORMAP)
        gui.set_image(pixels)
        gui.show()

if __name__ == '__main__':
    gen_static_mandelbrot_set()
    mandelbrot_viewer()