import numpy as np
import time
import os
import sys
import math
from collections import deque
import colorsys
import random

def get_colored_char(char, hue):
    # Преобразуем HSV в RGB
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1.0, 1.0)]
    return f"\033[38;2;{r};{g};{b}m{char}\033[0m"

def create_heart_points(scale=15, num_points=500, num_layers=30):
    t = np.linspace(0, 2*np.pi, num_points)
    x = 16 * np.sin(t)**3
    y = -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
    z = np.zeros_like(x)

    points = []
    for i in range(num_layers):
        factor = 1 - i/num_layers
        layer_x = factor * x
        layer_y = factor * y
        layer_z = np.full_like(x, i/3)
        points.extend(zip(layer_x, layer_y, layer_z))

    return scale * np.array(points)

def rotate_points(points, angle_x, angle_y, angle_z):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])
    
    return np.dot(points, Rz.T).dot(Ry.T).dot(Rx.T)

def draw_heart(points, width=80, height=40, shading_chars=" .:!*oe%&#@jhbkvnaihnvaqupanmbbnzsx48918046`4", time_val=0):
    screen = np.full((height, width), ' ', dtype=object)
    z_buffer = np.full((height, width), float('-inf'))
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x = (x / np.max(np.abs(x)) * (width//4) + width//2).astype(int)
    y = (y / np.max(np.abs(y)) * (height//4) + height//2).astype(int)
    
    mask = (0 <= x) & (x < width) & (0 <= y) & (y < height)
    x, y, z = x[mask], y[mask], z[mask]
    
    intensity = ((z - np.min(z)) / (np.max(z) - np.min(z)) * (len(shading_chars) - 1)).astype(int)
    
    for xi, yi, zi, char_index in zip(x, y, z, intensity):
        if zi > z_buffer[yi, xi]:
            z_buffer[yi, xi] = zi
            hue = (time_val + zi / np.max(z)) % 1.0
            screen[yi, xi] = get_colored_char(shading_chars[char_index], hue)
    
    return '\n'.join(''.join(row) for row in screen)

def pulsating_effect(time):
    return 1 + 0.1 * math.sin(time * 5)

def rainbow_effect(time):
    return (time * 0.1) % 1.0

def main():
    heart_points = create_heart_points(num_points=1000, num_layers=40)
    angle_x = angle_y = angle_z = 0
    
    print('\033[2J')
    print('\033[?25l')  # Hide cursor
    
    fps_counter = deque(maxlen=30)
    start_time = time.time()
    
    try:
        while True:
            frame_start = time.time()
            current_time = time.time() - start_time
            
            # Комбинируем эффекты пульсации и вращения
            scale = pulsating_effect(current_time)
            scaled_points = heart_points * scale
            rotated_points = rotate_points(scaled_points, angle_x, angle_y, angle_z)
            
            # Добавляем радужный эффект
            frame = draw_heart(rotated_points, time_val=rainbow_effect(current_time))
            
            fps = len(fps_counter) / (time.time() - fps_counter[0]) if fps_counter else 0
            
            status_line = f"\033[1mTime: {current_time:.2f}s | FPS: {fps:.2f} | Press Ctrl+C to exit\033[0m"
            frame_with_status = frame + "\n" + status_line
            
            sys.stdout.write('\033[H' + frame_with_status)
            sys.stdout.flush()
            
            angle_y += 0.1
            angle_x += 0.05
            angle_z += 0.03
            
            frame_end = time.time()
            frame_time = frame_end - frame_start
            fps_counter.append(frame_end)
            
            if frame_time < 0.03:
                time.sleep(0.03 - frame_time)
            
    except KeyboardInterrupt:
        print('\033[?25h')
        print("\nProgram terminated")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\033[?25h')
        print("\nProgram terminated")