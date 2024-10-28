import numpy as np
import time
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def create_heart_points(scale=15):
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = scale * np.outer(np.cos(u), np.sin(v))
    y = scale * np.outer(np.sin(u), np.sin(v))
    z = scale * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Применяем формулу сердца
    x = x * (1 - 0.8 * np.abs(z) / scale)
    y = y * (1 - 0.8 * np.abs(z) / scale) - 0.3 * z
    z = z * 0.8
    
    return np.column_stack((x.flatten(), y.flatten(), z.flatten()))

def rotate_points(points, angle):
    rotation_matrix = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    return np.dot(points, rotation_matrix.T)

def draw_heart(points, width=80, height=40):
    screen = [[' ' for _ in range(width)] for _ in range(height)]
    z_buffer = [[float('-inf') for _ in range(width)] for _ in range(height)]
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    x = (x / np.max(np.abs(x)) * (width//3)) + width//2
    y = (y / np.max(np.abs(y)) * (height//2)) + height//2
    
    for xi, yi, zi in zip(x, y, z):
        xi, yi = int(xi), int(yi)
        if 0 <= xi < width and 0 <= yi < height:
            if zi > z_buffer[yi][xi]:
                z_buffer[yi][xi] = zi
                intensity = int((zi - np.min(z)) / (np.max(z) - np.min(z)) * 3)
                screen[yi][xi] = '░▒▓█'[intensity]
    
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