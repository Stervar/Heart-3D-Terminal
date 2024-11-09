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

def create_heart_points(scale=15, num_points=800, num_layers=30):
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
    # Прекомпиляция тригонометрических функций
    cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
    cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
    
    rotated = points.copy()
    
    # Поворот вокруг оси X
    temp_y = rotated[:, 1] * cos_x - rotated[:, 2] * sin_x
    temp_z = rotated[:, 1] * sin_x + rotated[:, 2] * cos_x
    rotated[:, 1] = temp_y
    rotated[:, 2] = temp_z
    
    # Поворот вокруг оси Y
    temp_x = rotated[:, 0] * cos_y + rotated[:, 2] * sin_y
    temp_z = -rotated[:, 0] * sin_y + rotated[:, 2] * cos_y
    rotated[:, 0] = temp_x
    rotated[:, 2] = temp_z
    
    # Поворот вокруг оси Z
    temp_x = rotated[:, 0] * cos_z - rotated[:, 1] * sin_z
    temp_y = rotated[:, 0] * sin_z + rotated[:, 1] * cos_z
    rotated[:, 0] = temp_x
    rotated[:, 1] = temp_y
    
    return rotated

def draw_heart(points, width=60, height=30, shading_chars=None, time_val=0):
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
        shading_chars = '.:-=+*#%@'

    # Проекция 3D точек на 2D экран
    x = (points[:, 0] / np.max(np.abs(points[:, 0])) * (width//4) + width//2).astype(int)
    y = (points[:, 1] / np.max(np.abs(points[:, 1])) * (height//4) + height//2).astype(int)
    z = points[:, 2]
    
    # Фильтрация точек в пределах экрана
    mask = (0 <= x) & (x < width) & (0 <= y) & (y < height)
    x, y, z = x[mask], y[mask], z[mask]
    
    # Инициализация экрана и z-буфера
    screen = np.full((height, width), ' ', dtype=object)
    z_buffer = np.full((height, width), float('-inf'))
    
    # Нормализация глубины для выбора символов
    z_normalized = (z - np.min(z)) / (np.max(z) - np.min(z))
    intensity = (z_normalized * (len(shading_chars) - 1)).astype(int)
    
    # Отрисовка точек с учетом глубины и цвета
    for xi, yi, zi, char_index in zip(x, y, z, intensity):
        if zi > z_buffer[yi, xi]:
            z_buffer[yi, xi] = zi
            hue = (time_val + zi / np.max(z)) % 1.0
            screen[yi, xi] = get_colored_char(shading_chars[char_index], hue)
    
    return '\n'.join(''.join(row) for row in screen)

def calculate_fps(fps_counter):
    """
    Расчет кадров в секунду (FPS)
    
    Args:
        fps_counter (deque): Очередь временных меток кадров
    
    Returns:
        float: Количество кадров в секунду
    """
    if len(fps_counter) < 2:
        return 0.0
    time_diff = fps_counter[-1] - fps_counter[0]
    if time_diff <= 0:
        return 0.0
    return len(fps_counter) / time_diff

def main():
    """
    Основная функция для запуска анимации сердца
    """
    heart_points = create_heart_points(num_points=800, num_layers=30)  # Генерация точек сердца
    angle_x = angle_y = angle_z = 0  # Начальные углы вращения
    
    print('\033[2J')  # Очистка экрана
    print('\033[?25l')  # Скрытие курсора
    
    fps_counter = deque(maxlen=30)  # Очередь для расчета FPS
    start_time = time.time()  # Запись времени начала
    
    try:
        while True:
            frame_start = time.time()  # Время начала кадра
            current_time = time.time() - start_time  # Текущее время анимации
            
            scale = 1 + 0.1 * math.sin(current_time * 5)  # Пульсация
            scaled_points = heart_points * scale  # Масштабирование точек сердца
            rotated_points = rotate_points(scaled_points, angle_x, angle_y, angle_z)  # Вращение точек
            
            frame = draw_heart(rotated_points, time_val=(current_time * 0.1) % 1.0)  # Отрисовка сердца
            
            current_time = time.time()
            fps_counter.append(current_time)  # Добавление временной метки в очередь
            fps = calculate_fps(fps_counter)  # Расчет FPS
            
            status_line = f"\033[1mFPS: {fps:.1f} | Press Ctrl+C to exit\033[0m"  # Строка состояния
            frame_with_status = frame + "\n" + status_line
            
            sys.stdout.write('\033[H' + frame_with_status)  # Обновление экрана
            sys.stdout.flush()  # Сброс буфера
            
            angle_y += 0.1  # Увеличение угла вращения по оси Y
            angle_x += 0.05  # Увеличение угла вращения по оси X
            angle_z += 0.03  # Увеличение угла вращения по оси Z
            
            frame_time = time.time() - frame_start  # Время отрисовки кадра
            if frame_time < 0.016:  # Целевые 60 FPS
                time.sleep(0.016 - frame_time)  # Задержка для достижения 60 FPS
            
    except KeyboardInterrupt:
        print('\033[?25h')  # Показать курсор
        print("\nProgram terminated")  # Завершение программы

if __name__ == "__main__":
    try:
        main()  # Запуск основной функции
    except KeyboardInterrupt:
        print('\033[?25h')  # Показать курсор
        print("\nProgram terminated")  # Завершение программы
    except Exception as e:
        print('\033[?25h')  # Показать курсор
        print(f"\nAn error occurred: {str(e)}")  # Сообщение об ошибке
    finally:
        print('\033[0m')  # Сброс цветовых настроек