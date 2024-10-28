import numpy as np
import time
import os
import sys
import math
from collections import deque
import colorsys
import random

def get_colored_char(char, hue):
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1.0, 1.0)]
    return f"\033[38;2;{r};{g};{b}m{char}\033[0m"

def create_heart_points(scale=15, num_points=800, num_layers=30):
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
    cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
    cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
    
    rotated = points.copy()
    
    # Rotation X
    temp_y = rotated[:, 1] * cos_x - rotated[:, 2] * sin_x
    temp_z = rotated[:, 1] * sin_x + rotated[:, 2] * cos_x
    rotated[:, 1] = temp_y
    rotated[:, 2] = temp_z
    
    # Rotation Y
    temp_x = rotated[:, 0] * cos_y + rotated[:, 2] * sin_y
    temp_z = -rotated[:, 0] * sin_y + rotated[:, 2] * cos_y
    rotated[:, 0] = temp_x
    rotated[:, 2] = temp_z
    
    # Rotation Z
    temp_x = rotated[:, 0] * cos_z - rotated[:, 1] * sin_z
    temp_y = rotated[:, 0] * sin_z + rotated[:, 1] * cos_z
    rotated[:, 0] = temp_x
    rotated[:, 1] = temp_y
    
    return rotated

def draw_heart(points, width=60, height=30, shading_chars=None, time_val=0):
    if shading_chars is None:
        shading_chars = '.:-=+*#%@'

    x = (points[:, 0] / np.max(np.abs(points[:, 0])) * (width//4) + width//2).astype(int)
    y = (points[:, 1] / np.max(np.abs(points[:, 1])) * (height//4) + height//2).astype(int)
    z = points[:, 2]
    
    mask = (0 <= x) & (x < width) & (0 <= y) & (y < height)
    x, y, z = x[mask], y[mask], z[mask]
    
    screen = np.full((height, width), ' ', dtype=object)
    z_buffer = np.full((height, width), float('-inf'))
    
    z_normalized = (z - np.min(z)) / (np.max(z) - np.min(z))
    intensity = (z_normalized * (len(shading_chars) - 1)).astype(int)
    
    for xi, yi, zi, char_index in zip(x, y, z, intensity):
        if zi > z_buffer[yi, xi]:
            z_buffer[yi, xi] = zi
            hue = (time_val + zi / np.max(z)) % 1.0
            screen[yi, xi] = get_colored_char(shading_chars[char_index], hue)
    
    return '\n'.join(''.join(row) for row in screen)

def calculate_fps(fps_counter):
    if len(fps_counter) < 2:
        return 0.0
    time_diff = fps_counter[-1] - fps_counter[0]
    if time_diff <= 0:
        return 0.0
    return len(fps_counter) / time_diff

def main():
    heart_points = create_heart_points(num_points=800, num_layers=30)
    angle_x = angle_y = angle_z = 0
    
    print('\033[2J')
    print('\033[?25l')
    
    fps_counter = deque(maxlen=30)
    start_time = time.time()
    
    try:
        while True:
            frame_start = time.time()
            current_time = time.time() - start_time
            
            scale = 1 + 0.1 * math.sin(current_time * 5)
            scaled_points = heart_points * scale
            rotated_points = rotate_points(scaled_points, angle_x, angle_y, angle_z)
            
            frame = draw_heart(rotated_points, time_val=(current_time * 0.1) % 1.0)
            
            current_time = time.time()
            fps_counter.append(current_time)
            fps = calculate_fps(fps_counter)
            
            status_line = f"\033[1mFPS: {fps:.1f} | Press Ctrl+C to exit\033[0m"
            frame_with_status = frame + "\n" + status_line
            
            sys.stdout.write('\033[H' + frame_with_status)
            sys.stdout.flush()
            
            angle_y += 0.1
            angle_x += 0.05
            angle_z += 0.03
            
            frame_time = time.time() - frame_start
            if frame_time < 0.016:  # Целевые 60 FPS
                time.sleep(0.016 - frame_time)
            
    except KeyboardInterrupt:
        print('\033[?25h')
        print("\nProgram terminated")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\033[?25h')
        print("\nProgram terminated")