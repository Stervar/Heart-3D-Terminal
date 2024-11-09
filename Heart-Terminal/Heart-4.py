# Импорт необходимых библиотек для создания 3D сердца в терминале
import numpy as np      # Математические операции и работа с массивами
import time             # Работа со временем и задержками
import os               # Работа с операционной системой
from math import sin, cos, pi  # Математические функции для вращения

def clear_screen():
    """
    Очистка экрана с учетом операционной системы
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def create_heart_points(scale=40):
    """
    Генерация точек для создания 3D модели сердца с высокой детализацией
    
    Args:
        scale (int): Масштаб сердца
    
    Returns:
        np.array: Массив точек сердца
    """
    # Создание сетки точек с использованием сферических координат
    u = np.linspace(0, 2*np.pi, 300)  # Азимутальный угол
    v = np.linspace(0, np.pi, 150)    # Полярный угол
    
    # Базовые координаты сферы
    x = scale * np.outer(np.cos(u), np.sin(v))
    y = scale * np.outer(np.sin(u), np.sin(v))
    z = scale * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Деформация сферы для создания формы сердца
    x = x * (1 - 0.8 * np.abs(z) / scale)
    y = y * (1 - 0.8 * np.abs(z) / scale) - 0.3 * z
    z = z * 0.8
    
    # Добавление случайных возмущений для естественности
    x += np.random.normal(0, 0.3, x.shape)
    y += np.random.normal(0, 0.3, y.shape)
    
    return np.column_stack((x.flatten(), y.flatten(), z.flatten()))

def create_shading_chars(num_levels=30):
    """
    Создание набора символов для плавного градиента затенения
    
    Args:
        num_levels (int): Количество уровней затенения
    
    Returns:
        str: Строка символов для отображения
    """
    # Расширенный набор символов для более точного представления глубины
    basic_chars = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    
    # Повторение символов при недостаточном количестве
    if len(basic_chars) < num_levels:
        basic_chars = basic_chars * (num_levels // len(basic_chars) + 1)
    
    return basic_chars[:num_levels]

def rotate_points(points, angle_x, angle_y, angle_z):
    """
    Вращение точек в 3D пространстве вокруг всех осей
    
    Args:
        points (np.array): Массив точек
        angle_x (float): Угол поворота вокруг оси X
        angle_y (float): Угол поворота вокруг оси Y
        angle_z (float): Угол поворота вокруг оси Z
    
    Returns:
        np.array: Повернутые точки
    """
    # Матрицы поворота для каждой оси
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
    
    # Последовательное применение матриц поворота
    rotated = np.dot(points, rotation_x.T)
    rotated = np.dot(rotated, rotation_y.T)
    rotated = np.dot(rotated, rotation_z.T)
    
    return rotated

def calculate_lighting(point, light_source):
    """
    Расчет интенсивности освещения точки
    
    Args:
        point (np.array): Координаты точки
        light_source (np.array): Координаты источника света
    
    Returns:
        float: Интенсивность освещения
    """
    # Нормализация вектора точки и направления света
    normal = point / np.linalg.norm(point)
    light_dir = light_source / np.linalg.norm(light_source)
    
    # Расчет косинуса угла между вектором нормали и направлением света
    return np.dot(normal, light_dir)

def draw_heart(points, width=150, height=75):
    """
    Отрисовка сердца с использованием символов, z-буфера и освещения
    
    Args:
        points (np.array): Точки сердца
        width (int): Ширина экрана
        height (int): Высота экрана
    """
    # Инициализация экрана и z-буфера
    screen = [[' ' for _ in range(width)] for _ in range(height)]
    z_buffer = [[float('-inf') for _ in range(width)] for _ in range(height)]
    
    # Создание набора символов для затенения
    shading_chars = create_shading_chars(30)
    
    # Позиция источника света
    light_source = np.array([0, 0, 100])
    
    # Извлечение координат точек
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Масштабирование и центрирование точек
    x = (x / np.max(np.abs(x)) * (width//2.2)) + width//2
    y = (y / np.max(np.abs(y)) * (height//2.2)) + height//2
    
    # Отрисовка точек с учетом глубины и освещения
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        xi, yi = int(xi), int(yi)
        if 0 <= xi < width and 0 <= yi < height:
            if zi > z_buffer[yi][xi]:
                z_buffer[yi][xi] = zi
                
                # Расчет и нормализация интенсивности освещения
                light_intensity = calculate_lighting(points[i], light_source)
                light_intensity = (light_intensity + 1) / 2
                
                # Выбор символа на основе интенсивности освещения
                char_index = int(light_intensity * (len(shading_chars) - 1))
                char_index = max(0, min(char_index, len(shading_chars) - 1))
                
                screen[yi][xi] = shading_chars[char_index]
    
    # Вывод к ода с использованием двойного буфера для уменьшения мерцания
    frame = '\n'.join(''.join(row) for row in screen)
    print('\033[H' + frame)

def main():
    """
    Основная функция для запуска анимации сердца
    """
    # Генерация точек сердца
    heart_points = create_heart_points()
    angle_x = angle_y = angle_z = 0
    
    # Настройка терминала для быстрого вывода
    print('\033[2J')      # Очистка экрана
    print('\033[?25l')    # Скрытие курсора
    
    try:
        while True:
            # Вращение точек сердца
            rotated_points = rotate_points(heart_points, angle_x, angle_y, angle_z)
            draw_heart(rotated_points)
            
            # Обновление углов вращения
            angle_y += 0.15  # Основное вращение
            angle_x += 0.02  # Легкое покачивание
            angle_z += 0.01  # Небольшое кручение
            
            time.sleep(0.005)  # Минимальная задержка для плавности
            
    except KeyboardInterrupt:
        print('\033[?25h')  # Показать курсор
        print("\nПрограмма завершена")

if __name__ == "__main__":
    try:
        main()  # Запуск основной функции
    except KeyboardInterrupt:
        print('\033[?25h')  # Показать курсор
        print("\nПрограмма завершена")