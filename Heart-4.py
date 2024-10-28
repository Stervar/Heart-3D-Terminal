import numpy as np
import time
import os
from math import sin, cos, pi

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def create_heart_points(scale=40):
    # Увеличиваем количество точек для большей детализации
    u = np.linspace(0, 2*np.pi, 300)
    v = np.linspace(0, np.pi, 150)
    x = scale * np.outer(np.cos(u), np.sin(v))
    y = scale * np.outer(np.sin(u), np.sin(v))
    z = scale * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Улучшенная формула сердца с дополнительными деталями
    x = x * (1 - 0.8 * np.abs(z) / scale)
    y = y * (1 - 0.8 * np.abs(z) / scale) - 0.3 * z
    z = z * 0.8
    
    # Добавляем небольшие детали для более реалистичной формы
    x += np.random.normal(0, 0.3, x.shape)
    y += np.random.normal(0, 0.3, y.shape)
    
    return np.column_stack((x.flatten(), y.flatten(), z.flatten()))

def create_shading_chars(num_levels=30):
    # Создаем расширенный набор символов для более плавного градиента
    basic_chars = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    if len(basic_chars) < num_levels:
        basic_chars = basic_chars * (num_levels // len(basic_chars) + 1)
    return basic_chars[:num_levels]

def rotate_points(points, angle_x, angle_y, angle_z):
    # Улучшенная функция вращения с поддержкой всех осей
    rotation_x = np.array([
        [1, 0, 0],
        [0, cos(angle_x), -sin(angle_x)],
        [0, sin(angle_x), cos(angle_x)]
    ])
    
    rotation_y = np.array([
        [cos(angle_y), 0, sin(angle_y)],
        [0, 1, 0],
        [-sin(angle_y), 0, cos(angle_y)]
    ])
    
    rotation_z = np.array([
        [cos(angle_z), -sin(angle_z), 0],
        [sin(angle_z), cos(angle_z), 0],
        [0, 0, 1]
    ])
    
    rotated = np.dot(points, rotation_x.T)
    rotated = np.dot(rotated, rotation_y.T)
    rotated = np.dot(rotated, rotation_z.T)
    return rotated

def calculate_lighting(point, light_source):
    # Добавляем простое освещение
    normal = point / np.linalg.norm(point)
    light_dir = light_source / np.linalg.norm(light_source)
    return np.dot(normal, light_dir)

def draw_heart(points, width=150, height=75):
    screen = [[' ' for _ in range(width)] for _ in range(height)]
    z_buffer = [[float('-inf') for _ in range(width)] for _ in range(height)]
    
    # Создаем расширенный набор символов для отображения
    shading_chars = create_shading_chars(30)
    
    # Позиция источника света
    light_source = np.array([0, 0, 100])
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Улучшенное масштабирование для лучшего заполнения экрана
    x = (x / np.max(np.abs(x)) * (width//2.2)) + width//2
    y = (y / np.max(np.abs(y)) * (height//2.2)) + height//2
    
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        xi, yi = int(xi), int(yi)
        if 0 <= xi < width and 0 <= yi < height:
            if zi > z_buffer[yi][xi]:
                z_buffer[yi][xi] = zi
                
                # Рассчитываем освещение
                light_intensity = calculate_lighting(points[i], light_source)
                light_intensity = (light_intensity + 1) / 2  # Нормализуем к [0,1]
                
                # Применяем освещение к индексу символа
                char_index = int(light_intensity * (len(shading_chars) - 1))
                char_index = max(0, min(char_index, len(shading_chars) - 1))
                
                screen[yi][xi] = shading_chars[char_index]
    
    # Выводим с двойным буфером для уменьшения мерцания
    frame = '\n'.join(''.join(row) for row in screen)
    print('\033[H' + frame)

def main():
    heart_points = create_heart_points()
    angle_x = angle_y = angle_z = 0
    
    # Настройка терминала для быстрого вывода
    print('\033[2J')
    print('\033[?25l')  # Скрываем курсор
    
    try:
        while True:
            rotated_points = rotate_points(heart_points, angle_x, angle_y, angle_z)
            draw_heart(rotated_points)
            
            # Комплексное вращение
            angle_y += 0.15  # Основное вращение
            angle_x += 0.02  # Легкое покачивание
            angle_z += 0.01  # Небольшое кручение
            
            time.sleep(0.005)  # Минимальная задержка для плавности
            
    except KeyboardInterrupt:
        print('\033[?25h')  # Показываем курсор обратно
        print("\nПрограмма завершена")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\033[?25h')
        print("\nПрограмма завершена")
    