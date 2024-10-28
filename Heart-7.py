import numpy as np
import time
import os
import sys
import math
from collections import deque

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

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

def draw_heart(points, width=80, height=40, shading_chars=" .:!*oe%&#@"):
    screen = np.full((height, width), ' ', dtype=str)
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
            screen[yi, xi] = shading_chars[char_index]
    
    return '\n'.join(''.join(row) for row in screen)

def pulsating_effect(time):
    return 1 + 0.1 * math.sin(time * 5)

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
            
            scale = pulsating_effect(time.time() - start_time)
            scaled_points = heart_points * scale
            rotated_points = rotate_points(scaled_points, angle_x, angle_y, angle_z)
            
            frame = draw_heart(rotated_points)
            
            elapsed_time = time.time() - start_time
            fps = len(fps_counter) / (time.time() - fps_counter[0]) if fps_counter else 0
            
            status_line = f"Time: {elapsed_time:.2f}s | FPS: {fps:.2f} | Press Ctrl+C to exit"
            frame_with_status = frame + "\n" + status_line
            
            sys.stdout.write('\033[H' + frame_with_status)
            sys.stdout.flush()
            
            angle_y += 0.1
            angle_x += 0.05
            angle_z += 0.03
            
            frame_end = time.time()
            frame_time = frame_end - frame_start
            fps_counter.append(frame_end)
            
            if frame_time < 0.03:  # Trying to achieve around 30 FPS
                time.sleep(0.03 - frame_time)
            
    except KeyboardInterrupt:
        print('\033[?25h')  # Show cursor
        print("\nProgram terminated")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\033[?25h')
        print("\nProgram terminated")