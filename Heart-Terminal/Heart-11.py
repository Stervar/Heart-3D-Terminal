# Импорт необходимых библиотек для создания анимированного цветного 3D сердца в терминале
import numpy as np      # Математические операции и работа с массивами
import time             # Работа со временем и задержками
import sys              # Системные операции
import math             # Математические функции
from collections import deque  # Эффективная работа с очередью
import colorsys         # Преобразование цветов

def rotate_points(points, angle_y):
    """
    Функция поворота точек вокруг вертикальной оси (Y)
    
    Args:
        points (np.array): Массив точек для поворота
        angle_y (float): Угол поворота вокруг оси Y
    
    Returns:
        np.array: Повернутые точки
    """
    # Матрица поворота вокруг оси Y
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    
    return np.dot(points, Ry.T)

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

def get_colored_char(char, hue, saturation=1.0, value=1.0):
    """
    Преобразование символа в цветной с использованием HSV цветовой модели
    
    Args:
        char (str): Символ для окраски
        hue (float): Цветовой тон (0-1)
        saturation (float): Насыщенность (0-1)
        value (float): Яркость (0-1)
    
    Returns:
        str: Цветной символ с ANSI-кодировкой
    """
    # Ограничение значений в диапазоне 0-1
    hue = max(0.0, min(1.0, hue))
    saturation = max(0.0, min(1.0, saturation))
    value = max(0.0, min(1.0, value))
    
    # Преобразование HSV в RGB
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value)]
    return f"\033[38;2;{r};{g};{b}m{char}\033[0m"

def create_heart_points(scale=5, num_points=2000, num_layers=40):
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
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    z = np.zeros_like(x)

    points = []
    # Создание слоев сердца
    for i in range(num_layers):
        factor = 1 - i/num_layers
        layer_x = factor * x
        layer_y = factor * y
        layer_z = np.full_like(x, -i/3)
        points.extend(zip(layer_x, layer_y, layer_z))

    return scale * np.array(points)

def draw_heart(points, width=120, height=60, time_val=0):
    """
    Отрисовка сердца с использованием символов, z-буфера и цветов
    
    Args:
        points (np.array): Точки сердца
        width (int): Ширина экрана
        height (int): Высота экрана
        time_val (float): Временное значение для анимации цвета
    
    Returns:
        str: Строка с отрисованным сердцем
    """
    # Набор символов для различной интенсивности
    shading_chars = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    
    # Проекция 3D точек на 2D экран
    x = (points[:, 0] / np.max(np.abs(points[:, 0])) * (width//2) + width//2).astype(int)
    y = (points[:, 1] / np.max(np.abs(points[:, 1])) * (height//2) + height//2).astype(int)
    z = points[:, 2]
    
    # Фильтрация точек в пределах экрана
    mask = (0 <= x) & (x < width) & (0 <= y) & (y < height)
    x, y, z = x[mask], y[mask], z[mask]
    
    # Инициализация экрана и z-буфера
    screen = np.full((height, width), ' ', dtype=object)
    z_buffer = np.full((height, width), float('-inf'))
    
    if len(z) > 0:
        # Нормализация глубины для выбора символов и цвета
        z_min, z_max = np.min(z), np.max(z)
        if z_max > z_min:
            z_normalized = (z - z_min) / (z_max - z_min)
            intensity = (z_normalized * (len(shading_chars) - 1)).astype(int)
            
            # Отрисовка точек с учетом глубины и цвета
            for xi, yi, zi, char_index in zip(x, y, z, intensity):
                if zi > z_buffer[yi, xi]:
                    z_buffer[yi, xi] = zi
                    z_factor = (zi - z_min) / (z_max - z_min) if z_max > z_min else 0
                    hue = (time_val + z_factor) % 1.0
                    saturation = 0.8 + 0.2 * z_factor
                    value = 0.7 + 0.3 * z_factor
                    screen[yi, xi] = get_colored_char(shading_chars[char_index], hue, saturation, value)
    
    return '\n'.join(''.join(row) for row in screen)

def pulsating_effect(time):
    """
    Создание пульсирующего эффекта для сердца
    
    Args:
        time (float): Временное значение для расчета пульсации
    
    Returns:
        float: Масштаб для пульсации
    """
    return 1 + 0.05 * math.sin( time * 5)

def main():
    """
    Основная функция для запуска анимации сердца
    """
    heart_points = create_heart_points(scale=10)  # Генерация точек сердца
    angle_y = 0  # Начальный угол вращения вокруг оси Y
    
    print('\033[2J')  # Очистка экрана
    print('\033[?25l')  # Скрытие курсора
    
    fps_counter = deque(maxlen=30)  # Очередь для расчета FPS
    start_time = time.time()  # Запись времени начала
    
    try:
        while True:
            frame_start = time.time()  # Время начала кадра
            current_time = frame_start - start_time  # Текущее время анимации
            
            scale = pulsating_effect(current_time)  # Расчет масштаба для пульсации
            scaled_points = heart_points * scale  # Масштабирование точек сердца
            rotated_points = rotate_points(scaled_points, angle_y)  # Вращение точек
            
            rotated_points[:, 1] *= -1  # Инверсия оси Y
            
            frame = draw_heart(rotated_points, time_val=(current_time * 0.1) % 1.0)  # Отрисовка сердца
            
            fps_counter.append(time.time())  # Добавление временной метки в очередь
            fps = calculate_fps(fps_counter)  # Расчет FPS
            
            status_line = f"\033[1mFPS: {fps:.1f} | Time: {current_time:.2f}s | Press Ctrl+C to exit\033[0m"  # Строка состояния
            frame_with_status = frame + "\n" + status_line
            
            # Центрирование сердца в терминале
            terminal_width = 120
            frame_width = len(frame.split('\n')[0])
            padding = ' ' * ((terminal_width - frame_width) // 2)
            frame_with_status = padding + frame_with_status + padding
            
            sys.stdout.write('\033[H' + frame_with_status)  # Обновление экрана
            sys.stdout.flush()  # Сброс буфера
            
            angle_y += 0.05  # Вращение только по горизонтали
            
            frame_time = time.time() - frame_start  # Время отрисовки кадра
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
        print("\nProgram terminated")  # Завершение программы
    except Exception as e:
        print('\033[?25h')  # Показать курсор
        print(f"\nAn error occurred: {str(e)}")  # Сообщение об ошибке
    finally:
        print('\033[0m')  # Сброс цветовых настроек