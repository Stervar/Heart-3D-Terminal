import numpy as np
import time
import sys
import math
from collections import deque
import colorsys

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
    R = np.dot(Rz, np.dot(Ry, Rx))
    return np.dot(points, R.T)

def calculate_fps(fps_counter):
    if len(fps_counter) < 2:
        return 0.0
    time_diff = fps_counter[-1] - fps_counter[0]
    if time_diff <= 0:
        return 0.0
    return len(fps_counter) / time_diff

def get_colored_char(char, hue, saturation=1.0, value=1.0):
    hue = max(0.0, min(1.0, hue))
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value)]
    return f"\033[38;2;{r};{g};{b}m{char}\033[0m"

def create_heart_points(scale=5, num_points=2000, num_layers=50):
    t = np.linspace(0, 2*np.pi, num_points)
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    z = np.zeros_like(x)

    points = []
    for i in range(num_layers):
        factor = 1 - (i/num_layers)**1.2
        layer_x = factor * x
        layer_y = factor * y
        layer_z = np.full_like(x, -i/1.2)
        points.extend(zip(layer_x, layer_y, layer_z))

    for _ in range(num_points):
        r = np.random.random() * 0.9
        theta = np.random.random() * 2 * np.pi
        phi = np.random.random() * np.pi
        x = r * 16 * np.sin(theta)**3 * np.sin(phi)
        y = r * (13 * np.cos(theta) - 5 * np.cos(2*theta) - 2 * np.cos(3*theta) - np.cos(4*theta)) * np.sin(phi)
        z = r * 20 * np.cos(phi)
        points.append((x, y, z))

    return scale * np.array(points)

def draw_heart(points, width=100, height=50, time_val=0):
    shading_chars = ".:!*OQ#•●~`08'°></|Оо⊖⊘⊙⊚⊛⊜⊝◉○◌◍◎●◐◑◒⬬⬭⬮⬯"
    
    x = (points[:, 0] / np.max(np.abs(points[:, 0])) * (width//2) + width//2).astype(int)
    y = (points[:, 1] / np.max(np.abs(points[:, 1])) * (height//2) + height//2).astype(int)
    z = points[:, 2]
    
    mask = (0 <= x) & (x < width) & (0 <= y) & (y < height)
    x, y, z = x[mask], y[mask], z[mask]
    
    screen = np.full((height, width), ' ', dtype=object)
    z_buffer = np.full((height, width), float('-inf'))
    
    if len(z) > 0:
        z_min, z_max = np.min(z), np.max(z)
        if z_max > z_min:
            z_normalized = (z - z_min) / (z_max - z_min)
            intensity = (z_normalized * (len(shading_chars) - 1)).astype(int)
            
            for xi, yi, zi, char_index in zip(x, y, z, intensity):
                if zi > z_buffer[yi, xi]:
                    z_buffer[yi, xi] = zi
                    z_factor = (zi - z_min) / (z_max - z_min) if z_max > z_min else 0
                    hue = (time_val + z_factor) % 1.0
                    screen[yi, xi] = get_colored_char(shading_chars[char_index], hue, 1.0, 1.0)
    
    return '\n'.join(''.join(row) for row in screen)

def pulsating_effect(time):
    return 1 + 0.08 * math.sin(time * 1.5)

def main():
    heart_points = create_heart_points(scale=10)
    angle_x, angle_y, angle_z = 0, 0, 0
    
    print('\033[2J')
    print('\033[?25l')
    
    fps_counter = deque(maxlen=30)
    start_time = time.time()
    
    try:
        while True:
            frame_start = time.time()
            current_time = frame_start - start_time
            
            scale = pulsating_effect(current_time)
            scaled_points = heart_points * scale
            rotated_points = rotate_points(scaled_points, angle_x, angle_y, angle_z)
            
            rotated_points[:, 1] *= -1
            
            frame = draw_heart(rotated_points, width=100, height=50, time_val=(current_time * 0.1) % 1.0)
            
            fps_counter.append(time.time())
            fps = calculate_fps(fps_counter)
            
            status_line = f"\033[1mFPS: {fps:.1f} | Press Ctrl+C to exit\033[0m"
            frame_with_status = frame + "\n" + status_line
            
            terminal_width = 100
            frame_width = len(frame.split('\n')[0])
            padding = ' ' * ((terminal_width - frame_width) // 2)
            frame_with_status = padding + frame_with_status.replace('\n', f'\n{padding}')
            
            sys.stdout.write('\033[H' + frame_with_status)
            sys.stdout.flush()
            
            angle_y += 0.04
            angle_x = 0.2 * math.sin(current_time * 0.5)
            angle_z = 0.1 * math.cos(current_time * 0.3)
            
            frame_time = time.time() - frame_start
            if frame_time < 0.033:
                time.sleep(0.033 - frame_time)
            
    except KeyboardInterrupt:
        print('\033[?25h')
        print("\nProgram terminated")

if __name__ == "__main__":
    try:
        sys.stdout.write('\x1b[8 ;50;100t')
        print('\033[2J\033[H')
        main()
    except KeyboardInterrupt:
        print('\033[?25h')
        print("\nProgram terminated")
    except Exception as e:
        print('\033[?25h')
        print(f"\nAn error occurred: {str(e)}")
    finally:
        print('\033[0m')
        
        