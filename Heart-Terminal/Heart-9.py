# Импорт необходимых библиотек для создания анимированного цветного 3D сердца в терминале
import numpy as np      # Математические операции и работа с массивами
import time             # Работа со временем и задержками
import os               # Работа с операционной системой
import sys              # Системные операции
import math             # Математические функции
from collections import deque  # Эффективная работа с очередью
import colorsys         # Преобразование цветов
import random           # Генерация случайных чисел

def get_colored_char(char, hue):
    """
    Преобразование символа в цветной с использованием HSV цветовой модели
    
    Args:
        char (str): Символ для окраски
        hue (float): Цветовой тон (0-1)
    
    Returns:
        str: Цветной символ с ANSI-кодировкой
    """
    # Преобразование HSV в RGB
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1.0, 1.0)]
    return f"\033[38;2;{r};{g};{b}m{char}\033[0m"

def create_heart_points(scale=15, num_points=500, num_layers=30):
    """
    Генерация точек для создания 3D модели сердца
    
    Args:
        scale (int): Масштаб сердца
        num_points (int): Количество точек
        num_layers (int): Количество слоев
    
    Returns:
        np.array: Массив точек сердца
    """
    # Параметрическое уравнение сердца
    t = np.linspace(0, 2*np.pi, num_points)
    x = 16 * np.sin(t)**3
    y = -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
    z = np.zeros_like(x)

    points = []
    # Создание слоев сердца
    for i in range(num_layers):
        factor = 1 - i/num_layers
        layer_x = factor * x
        layer_y = factor * y
        layer_z = np.full_like(x, i/3)
        points.extend(zip(layer_x, layer_y, layer_z))

    return scale * np.array(points)

def rotate_points(points, angle_x, angle_y, angle_z):
    """
    Функция поворота точек вокруг трех осей (X, Y, Z)
    
    Args:
        points (np.array): Массив точек для поворота
        angle_x (float): Угол поворота вокруг оси X
        angle_y (float): Угол поворота вокруг оси Y
        angle_z (float): Угол поворота вокруг оси Z
    
    Returns:
        np.array: Повернутые точки
    """
    # Матрицы поворота вокруг каждой оси
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
    
    # Применение матриц поворота в определенном порядке
    return np.dot(points, Rz.T).dot(Ry.T).dot(Rx.T)

def draw_heart(points, width=80, height=40, shading_chars=None, time_val=0):
    """
    Отрисовка сердца с использованием символов, z-буфера и цветов
    
    Args:
        points (np.array): Точки сердца
        width (int): Ширина экрана
        height (int): Высота экрана
        shading_chars (list): Список символов для затенения
        time_val (float): Временное значение для анимации цвета
    
    Returns:
        str: Строка с отрисованным сердцем
    """
    # Набор символов для различной интенсивности
    if shading_chars is None:
        # Использование широкого диапазона символов ASCII
        shading_chars = ''.join(chr(i) for i in range(32, 127)) * 8

    # Инициализация экрана и z-буфера
    screen = np.full((height, width), ' ', dtype=object)
    z_buffer = np.full((height, width), float('-inf'))
    
    # Извлечение координат точек
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Проекция 3D точек на 2D экран
    x = (x / np.max(np.abs(x)) * (width//4) + width//2).astype(int)
    y = (y / np.max(np.abs(y)) * (height//4) + height//2).astype(int)
    
    # Фильтрация точек в пределах экрана
    mask = (0 <= x) & (x < width) & (0 <= y) & (y < height)
    x, y, z = x[mask], y[mask], z[mask]
    
    # Нормализация глубины для выбора символов
    intensity = ((z - np.min(z)) / (np.max(z) - np.min(z)) * (len(shading_chars) - 1)).astype(int)
    
    # Отрисовка точек с учетом глубины и цвета
    for xi, yi, zi, char_index in zip(x, y, z, intensity):
        if zi > z_buffer[yi, xi]:
            z_buffer[yi, xi] = zi
            # Создание плавного цветового перехода в зависимости от глубины
            hue = (time_val + zi / np.max(z)) % 1.0
            screen[yi, xi] = get_colored_char(shading_chars[char_index], hue)
    
    return '\n'.join(''.join(row) for row in screen)

def pulsating_effect(time):
    """
    Создание пульсирующего эффекта для сердца
    
    Args:
        time (float): Временное значение для расчета пульсации
    
    Returns:
        float: Масштаб для пульсации
    """
    return 1 + 0.1 * math.sin(time * 5)

def rainbow_effect(time):
    """
    Создание радужного эффекта для цветов
    
    Args:
        time (float): Временное значение для расчета цвета
    
    Returns:
        float: Значение оттенка
    """
    return (time * 0.1) % 1.0

def main():
 
    """
    Основная функция для запуска анимации сердца
    """
    heart_points = create_heart_points(num_points=1000, num_layers=40)  # Генерация точек сердца
    angle_x = angle_y = angle_z = 0  # Начальные углы вращения
    
    print('\033[2J')  # Очистка экрана
    print('\033[?25l')  # Скрытие курсора
    
    fps_counter = deque(maxlen=30)  # Очередь для расчета FPS
    start_time = time.time()  # Запись времени начала
    
    try:
        while True:
            frame_start = time.time()  # Время начала кадра
            current_time = time.time() - start_time  # Текущее время анимации
            
            scale = pulsating_effect(current_time)  # Получение масштаба для пульсации
            scaled_points = heart_points * scale  # Масштабирование точек сердца
            rotated_points = rotate_points(scaled_points, angle_x, angle_y, angle_z)  # Вращение точек
            
            frame = draw_heart(rotated_points, time_val=rainbow_effect(current_time))  # Отрисовка сердца
            
            # Расчет FPS
            fps = len(fps_counter) / (time.time() - fps_counter[0]) if fps_counter else 0
            
            # Формирование строки состояния
            status_line = f"\033[1mTime: {current_time:.2f}s | FPS: {fps:.2f} | Press Ctrl+C to exit\033[0m"
            frame_with_status = frame + "\n" + status_line
            
            sys.stdout.write('\033[H' + frame_with_status)  # Обновление экрана
            sys.stdout.flush()  # Сброс буфера
            
            # Обновление углов вращения
            angle_y += 0.1
            angle_x += 0.05
            angle_z += 0.03
            
            frame_end = time.time()  # Время окончания кадра
            frame_time = frame_end - frame_start  # Время отрисовки кадра
            fps_counter.append(frame_end)  # Добавление временной метки в очередь
            
            if frame_time < 0.03:  # Целевые 30 FPS
                time.sleep(0.03 - frame_time)  # Задержка для достижения 30 FPS
            
    except KeyboardInterrupt:
        print('\033[?25h')  # Показать курсор
        print("\nProgram terminated")  # Завершение программы

if __name__ == "__main__":
    try:
        main()  # Запуск основной функции
    except KeyboardInterrupt:
        print('\033[?25h')  # Показать курсор
        print("\nProgram terminated")  # Завершение ⬤