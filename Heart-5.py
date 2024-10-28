import numpy as np
import time
import os
from math import sin, cos, pi

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def create_heart_points(scale=15):
    # Уменьшаем количество точек для скорости
    t = np.linspace(0, 2*np.pi, 1000)
    x = 16 * np.sin(t)**3
    y = -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
    z = np.zeros_like(x)

    # Создаем меньше слоев для 3D эффекта
    for i in range(50):
        factor = 1 - i/50
        x = np.append(x, factor * 16 * np.sin(t)**3)
        y = np.append(y, factor * -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)))
        z = np.append(z, np.full_like(t, i/3))

    return scale * np.column_stack((x, y, z))

def rotate_points(points, angle_x, angle_y, angle_z):
    # Упрощенное вращение
    rotation = np.array([
        [cos(angle_y), 0, sin(angle_y)],
        [0, 1, 0],
        [-sin(angle_y), 0, cos(angle_y)]
    ])
    return np.dot(points, rotation.T)

def draw_heart(points, width=80, height=40):  # Уменьшен размер вывода
    screen = [[' ' for _ in range(width)] for _ in range(height)]
    z_buffer = [[float('-inf') for _ in range(width)] for _ in range(height)]
    
    # Упрощенный набор символов
    shading_chars = " .:!*oe%&#@"
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x = (x / np.max(np.abs(x)) * (width//4)) + width//2
    y = (y / np.max(np.abs(y)) * (height//4)) + height//2
    
    for xi, yi, zi in zip(x, y, z):
        xi, yi = int(xi), int(yi)
        if 0 <= xi < width and 0 <= yi < height:
            if zi > z_buffer[yi][xi]:
                z_buffer[yi][xi] = zi
                intensity = int((zi - np.min(z)) / (np.max(z) - np.min(z)) * (len(shading_chars) - 1))
                screen[yi][xi] = shading_chars[intensity]
    
    print('\033[H' + '\n'.join(''.join(row) for row in screen))

def main():
    heart_points = create_heart_points()
    angle_x = angle_y = angle_z = 0
    
    print('\033[2J')
    print('\033[?25l')
    
    try:
        while True:
            rotated_points = rotate_points(heart_points, angle_x, angle_y, angle_z)
            draw_heart(rotated_points)
            
            # Увеличена скорость вращения
            angle_y += 0.3
            angle_x += 0.1
            angle_z += 0.05
            
    except KeyboardInterrupt:
        print('\033[?25h')
        print("\nПрограмма завершена")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\033[?25h')
        print("\nПрограмма завершена")