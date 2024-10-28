import numpy as np
import time
import os
import sys

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def create_heart_points(scale=15):
    t = np.linspace(0, 2*np.pi, 500)  # Уменьшено количество точек
    x = 16 * np.sin(t)**3
    y = -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
    z = np.zeros_like(x)

    for i in range(25):  # Уменьшено количество слоев
        factor = 1 - i/25
        x = np.append(x, factor * 16 * np.sin(t)**3)
        y = np.append(y, factor * -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)))
        z = np.append(z, np.full_like(t, i/3))

    return scale * np.column_stack((x, y, z))

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

def draw_heart(points, width=80, height=40):
    screen = np.full((height, width), ' ', dtype=str)
    z_buffer = np.full((height, width), float('-inf'))
    
    shading_chars = " .:!*oe%&#@"
    
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
    
    sys.stdout.write('\033[H' + '\n'.join(''.join(row) for row in screen))
    sys.stdout.flush()

def main():
    heart_points = create_heart_points()
    angle_x = angle_y = angle_z = 0
    
    print('\033[2J')
    print('\033[?25l')
    
    try:
        while True:
            start_time = time.time()
            
            rotated_points = rotate_points(heart_points, angle_x, angle_y, angle_z)
            draw_heart(rotated_points)
            
            angle_y += 0.2
            angle_x += 0.1
            angle_z += 0.05
            
            elapsed_time = time.time() - start_time
            if elapsed_time < 0.03:  # Пытаемся достичь примерно 30 FPS
                time.sleep(0.03 - elapsed_time)
            
    except KeyboardInterrupt:
        print('\033[?25h')
        print("\nПрограмма завершена")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\033[?25h')
        print("\nПрограмма завершена")