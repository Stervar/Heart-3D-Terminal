import numpy as np
import time
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def create_heart_points(scale=10):
    t = np.linspace(0, 2*np.pi, 100)
    x = scale * 16 * np.sin(t)**3
    y = scale * (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
    z = np.zeros_like(x)
    return np.column_stack((x, y, z))

def rotate_points(points, angle):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    return np.dot(points, rotation_matrix)

def draw_heart(points, width=60, height=30):
    screen = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Проецируем 3D точки на 2D экран
    x = points[:, 0]
    y = points[:, 1]
    
    # Масштабируем и смещаем точки
    x = (x / np.max(np.abs(x)) * (width//3)) + width//2
    y = (y / np.max(np.abs(y)) * (height//3)) + height//2
    
    # Рисуем точки
    for xi, yi in zip(x, y):
        if 0 <= int(xi) < width and 0 <= int(yi) < height:
            screen[int(yi)][int(xi)] = '❤'
    
    # Выводим экран
    for row in screen:
        print(''.join(row))

def main():
    heart_points = create_heart_points()
    angle = 0
    
    while True:
        clear_screen()
        rotated_points = rotate_points(heart_points, angle)
        draw_heart(rotated_points)
        angle += 0.1
        time.sleep(0.05)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрограмма завершена")