import numpy as np  # Математические операции и работа с массивами
import time  # Работа со временем и задержками
import os  # Работа с операционной системой
import sys  # Системные операции
import math  # Математические функции

def create_heart_points(scale=15, num_points=800, num_layers=30):
    """
    Генерация точек для создания 3D модели сердца
    
    Args:
        scale (int): Масштаб сердца
        num_points (int): Количество точек
        num_layers (int): Количество слоев для объемности
    
    Returns:
        np.array: Массив точек сердца
    """
    # Параметрическое уравнение сердца
    t = np.linspace(0, 2*np.pi, num_points)
    x = 16 * np.sin(t)**3
    y = -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
    z = np.zeros_like(x)

    points = []
    # Создание послойной структуры сердца
    for i in range(num_layers):
        factor = 1 - i/num_layers
        layer_x = factor * x
        layer_y = factor * y
        layer_z = np.full_like(x, i/3)
        points.extend(zip(layer_x, layer_y, layer_z))

    return scale * np.array(points)

def vertical_motion(points, time_val):
    """
    Добавление вертикального движения к точкам
    
    Args:
        points (np.array): Массив точек
        time_val (float): Текущее время для анимации
    
    Returns:
        np.array: Точки с вертикальным смещением
    """
    # Синусоидальное вертикальное движение
    vertical_offset = 10 * np.sin(time_val * 2)
    
    moved_points = points.copy()
    moved_points[:, 1] += vertical_offset
    
    return moved_points

def rotate_points(points, angle_y):
    """
    Функция поворота точек вокруг оси Y
    
    Args:
        points (np.array): Массив точек
        angle_y (float): Угол поворота
    
    Returns:
        np.array: Повернутые точки
    """
    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
    
    rotated = points.copy()
    
    # Матрица поворота вокруг оси Y
    temp_x = rotated[:, 0] * cos_y + rotated[:, 2] * sin_y
    temp_z = -rotated[:, 0] * sin_y + rotated[:, 2] * cos_y
    rotated[:, 0] = temp_x
    rotated[:, 2] = temp_z
    
    return rotated

def draw_heart(points, width=60, height=30):
    """
    Отрисовка сердца с использованием символов для имитации глубины
    
    Args:
        points (np.array): Точки сердца
        width (int): Ширина экрана
        height (int): Высота экрана
    
    Returns:
        str: Строка с отрисованным сердцем
    """
    # Символы для имитации глубины
    shading_chars = '.:-=+*#%@'

    # Инициализация экрана и z-буфера
    screen = np.full((height, width), ' ', dtype=str)
    z_buffer = np.full((height, width), float('-inf'))
    
    # Проекция 3D точек на 2D экран
    x = (points[:, 0] / np.max(np.abs(points[:, 0])) * (width//4) + width//2).astype(int)
    y = (points[:, 1] / np.max(np.abs(points[:, 1])) * (height//4) + height//2).astype(int)
    z = points[:, 2]
    
    # Фильтрация точек в пределах экрана
    mask = (0 <= x) & (x < width) & (0 <= y) & (y < height)
    x, y, z = x[mask], y[mask], z[mask]
    
    # Нормализация глубины для выбора символов
    z_normalized = (z - np.min(z)) / (np.max(z) - np.min(z))
    intensity = (z_normalized * (len(shading_chars) - 1)).astype(int)
    
    # Отрисовка точек с учетом глубины
    for xi, yi, zi, char_index in zip(x, y, z, intensity):
        if zi > z_buffer[yi, xi]:
            z_buffer[yi, xi] = zi
            screen[yi, xi] = shading_chars[char_index]
    
    return '\n'.join(''.join(row) for row in screen)

def main():
    """
    Основная функция для запуска анимации сердца
    """
    # Очистка экрана
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Генерация точек сердца
    heart_points = create_heart_points(num_points=800, num_layers=30)
    angle_y = 0
    
    try:
        while True:
            # Получение текущего времени
            current_time = time.time()
            
            # Вертикальное движение и вращение
            moved_points = vertical_motion(heart_points, current_time)
            rotated_points = rotate_points(moved_points, angle_y)
            
            # Масштабирование с эффектом пульсации
            scale = 1 + 0.1 * math.sin(current_time * 5)
            scaled_points = rotated_points * scale
            
            # Отрисовка сердца
            frame = draw_heart(scaled_points)
            
            # Очистка экрана и вывод
            os.system('cls' if os.name == 'nt' else 'clear')
            print(frame)
            
            # Обновление угла поворота
            angle_y += 0.1
            
            # Задержка для контроля скорости анимации
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\nПрограмма завершена")

if __name__ == "__main__":
    main()