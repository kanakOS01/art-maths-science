"""
Game of Life written in 100 lines of Taichi
In memory of John Horton Conway (1937 - 2020)

https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/game_of_life.py
"""

import numpy as np
import taichi as ti

ti.init(arch=ti.gpu)


n = 64
cell_size = 8
img_size = n * cell_size

alive = ti.field(int, shape=(n, n))
count = ti.field(int, shape=(n, n))

@ti.func
def get_alive(i, j):
    return alive[i, j] if 0 <= i < n and 0 <= j < n else 0


@ti.func
def get_count(i, j):
    return (
        get_alive(i-1, j)
        + get_alive(i+1, j)
        + get_alive(i, j-1)
        + get_alive(i, j+1)
        + get_alive(i-1, j-1)
        + get_alive(i+1, j-1)
        + get_alive(i-1, j+1)
        + get_alive(i+1, j+1)
    )

B, S = [3], [2, 3]


@ti.func
def calc_rule(a, c):
    if a == 0:
        for t in ti.static(B):
            if c == t:
                a = 1
    elif a == 1:
        a = 0
        for t in ti.static(S):
            if c == t:
                a = 1
    return a


@ti.kernel
def run():
    for i, j in alive:
        count[i, j] = get_count(i, j)
    
    for i, j in alive:
        alive[i, j] = calc_rule(alive[i, j], count[i, j])


@ti.kernel
def init():
    for i, j in alive:
        if ti.random() > 0.8:
            alive[i, j] = 1
        else:
            alive[i, j] = 0

    
def main():
    gui = ti.GUI("Game of Life", (img_size, img_size))
    gui.fps_limit = 10
    
    print("[HINT] Press `r` to reset")
    print("[HINT] Press `SPACE` to pause")
    print("[HINT] Click LMB, RMB and drag to add alive/dead cells")

    
    init()
    paused = False
    while gui.running:
        for e in gui.get_events(gui.PRESS, gui.MOTION):
            if e.key == gui.ESCAPE:
                gui.running = False
            elif e.key == gui.SPACE:
                paused = not paused
            elif e.key == 'r':
                alive.fill(0)

        if gui.is_pressed(gui.LMB, gui.RMB):
            mx, my = gui.get_cursor_pos()
            alive[int(mx * n), int(my * n)] = gui.is_pressed(gui.LMB)
            paused = True
        
        if not paused:
            run()

        gui.set_image(ti.tools.imresize(alive, img_size).astype(np.uint8) * 255)
        gui.show()


if __name__ == "__main__":
    main()
