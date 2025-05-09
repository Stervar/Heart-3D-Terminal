# Импорт необходимых библиотек для создания анимированного 3D сердца в терминале
import numpy as np      # Математические операции и работа с массивами
import time             # Работа со временем и задержками
import sys              # Системные операции
import math             # Математические функции
from collections import deque  # Эффективная работа с очередью

def rotate_points(points, angle_y):
    """
    Функция поворота точек вокруг вертикальной оси (Y)
    
    Args:
        points (np.array): Массив точек для поворота
        angle_y (float): Угол поворота вокруг оси Y
    
    Returns:
        np.array: Повернутые точки
    """
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
    return len(fps_counter) / time_diff if time_diff > 0 else 0.0

def create_heart_points(scale=5, num_points=2000, num_layers=50):
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
        factor = 1 - (i/num_layers)**1.2
        layer_x = factor * x
        layer_y = factor * y
        layer_z = np.full_like(x, -i/1.2)
        points.extend(zip(layer_x, layer_y, layer_z))

    # Добавление случайных точек внутри объема сердца
    for _ in range(num_points):
        r = np.random.random() * 0.9
        theta = np.random.random() * 2 * np.pi
        phi = np.random.random() * np.pi
        x = r * 16 * np.sin(theta)**3 * np.sin(phi)
        y = r * (13 * np.cos(theta) - 5 * np.cos(2*theta) - 2 * np.cos(3*theta) - np.cos(4*theta)) * np.sin(phi)
        z = r * 20 * np.cos(phi)
        points.append((x, y, z))

    return scale * np.array(points)

def draw_heart(points, width=100, height=50):
    """
    Отрисовка сердца с использованием символов и z-буфера
    
    Args:
        points (np.array): Точки сердца
        width (int): Ширина экрана
        height (int): Высота экрана
    
    Returns:
        str: Строка с отрисованным сердцем
    """
    # Набор символов для различной интенсивности
    shading_chars = " .:-=+*#%@"
    
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
        # Нормализация глубины для выбора символов
        z_min, z_max = np.min(z), np.max(z)
        if z_max > z_min:
            z_normalized = (z - z_min) / (z_max - z_min)
            intensity = (z_normalized * (len(shading_chars) - 1)).astype(int)
            
            # Отрисовка точек с учетом глубины
            for xi, yi, zi, char_index in zip(x, y, z, intensity):
                if zi > z_buffer[yi, xi]:
                    z_buffer[yi, xi] = zi
                    screen[yi, xi] = shading_chars[char_index]
    
    return '\n'.join(''.join(row) for row in screen)

def pulsating_effect(time):
    """
    Создание пульсирующего эффекта для сердца
    
    Args:
        time (float): Временное значение для расчета пульсации
    
    Returns:
        float: Масштаб для пульсации
    """
    return 1 + 0.08 * math.sin(time * 1.5)

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
            
            rotated_points[:, 1] *= -1  # Инверсия оси Y для корректного отображения
            
            frame = draw_heart(rotated_points, width=100, height=50)  # Отрисовка сердца
            
            fps_counter.append(time.time())  # Добавление временной метки в очередь
            fps = calculate_fps(fps_counter)  # Расчет FPS
            
            status_line = f"FPS: {fps:.1f} | Press Ctrl+C to exit"  # Строка статуса
            frame_with_status = frame + "\n" + status_line  # Объединение кадра и строки статуса
            
            terminal_width = 100  # Ширина терминала
            frame_width = len(frame.split('\n')[0])  # Ширина кадра
            padding = ' ' * ((terminal_width - frame_width) // 2)  # Вычисление отступа
            frame_with_status = padding + frame_with_status.replace('\n', f'\n{padding}')  # Добавление отступа
            
            sys.stdout.write('\033[H' + frame_with_status)  # Вывод кадра в терминал
            sys.stdout.flush()  # Обновление вывода
            
            angle_y += 0.04  # Увеличение угла вращения
            
            frame_time = time.time() - frame_start  # Время отрисовки кадра
            if frame_time < 0.033:  # Если время меньше 33 мс
                time.sleep(0.033 - frame_time)  # Задержка для поддержания частоты кадров
            
    except KeyboardInterrupt:
        print('\033[?25h')  # Показ курсора
        print("\nProgram terminated")  # Сообщение о завершении программы

if __name__ == "__main__":
    try:
        sys.stdout.write('\x1b[8 ;50;100t')  # Установка размера терминала
        print('\033[2J\033[H')  # Очистка экрана и перемещение курсора в начало
        main()  # Запуск основной функции
    except KeyboardInterrupt:
        print('\033[?25h')  # Показ курсора
        print("\nProgram terminated")  # Сообщение о завершении программы
    except Exception as e:
        print('\033[?25h')  # Показ курсора
        print(f"\nAn error occurred: {str(e)}")  # Сообщение об ошибке
    finally:
        print('\033[0m')  # Сброс цветовых настроек терминала