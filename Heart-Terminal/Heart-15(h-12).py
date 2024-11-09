#Новый код с эффектом мерцания и более качвественным перелеванием


# Импорт необходимых библиотек
import numpy as np  # Библиотека для работы с многомерными массивами и математическими вычислениями
import time  # Модуль для работы со временем и создания задержек
import sys  # Модуль для взаимодействия с системой и терминалом
import math  # Математические функции (sin, cos и др.)
from collections import deque  # Двусторонняя очередь для эффективного подсчета FPS
import colorsys  # Модуль для преобразования цветовых пространств (HSV в RGB)

def rotate_points(points, angle_y):
    """
    Функция поворота точек вокруг оси Y в трехмерном пространстве
    
    Параметры:
    - points: массив трехмерных точек
    - angle_y: угол поворота вокруг оси Y
    
    Возвращает повернутые точки
    """
    # Создание матрицы поворота вокруг оси Y
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],     # Первая строка матрицы поворота
        [0, 1, 0],                                 # Вторая строка (ось Y не меняется)
        [-np.sin(angle_y), 0, np.cos(angle_y)]     # Третья строка матрицы поворота
    ])
    
    # Умножение точек на транспонированную матрицу поворота
    return np.dot(points, Ry.T)

def calculate_fps(fps_counter):
    """
    Расчет частоты кадров (FPS)
    
    Параметр:
    - fps_counter: очередь временных меток кадров
    
    Возвращает количество кадров в секунду
    """
    # Проверка наличия достаточного количества меток
    if len(fps_counter) < 2:
        return 0.0
    
    # Вычисление разницы времени между первой и последней меткой
    time_diff = fps_counter[-1] - fps_counter[0]
    
    # Защита от деления на ноль
    if time_diff <= 0:
        return 0.0

    # Расчет FPS: количество кадров / время
    return len(fps_counter) / time_diff

def get_colored_char(char, hue, saturation=1.0, value=1.0):
    """
    Преобразование символа в цветной символ с использованием цветового пространства HSV
    
    Параметры:
    - char: символ для окрашивания
    - hue: оттенок (0.0 - 1.0)
    - saturation: насыщенность (0.0 - 1.0)
    - value: яркость (0.0 - 1.0)
    
    Возвращает ANSI-последовательность для цветного символа
    """
    # Ограничение оттенка в диапазоне [0, 1]
    hue = max(0.0, min(1.0, hue))
    
    # Преобразование HSV в RGB и масштабирование до 255
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value)]
    
    # Формирование ANSI-последовательности для цветного символа
    return f"\033[38;2;{r};{g};{b}m{char}\033[0m"

def create_heart_points(scale=5, num_points=1000, num_layers=30):
    """
    Создание точек для формирования 3D-сердца
    
    Параметры:
    - scale: масштаб сердца
    - num_points: количество точек на контуре
    - num_layers: количество слоев для объемности
    
    Возвращает массив точек сердца
    """
    # Создание параметрического массива для генерации контура сердца
    t = np.linspace(0, 2*np.pi, num_points)
    
    # Параметрические уравнения сердца
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    z = np.zeros_like(x)

    points = []

    # Создание слоев для объемности сердца (нижняя часть)
    for i in range(num_layers):
        factor = 1 - i/num_layers
        layer_x = factor * x
        layer_y = factor * y
        layer_z = np.full_like(x, -i/2)
        points.extend(zip(layer_x, layer_y, layer_z))

    # Добавление внутренних точек для реалистичности
    for _ in range(num_points // 2):
        r = np.random.random() * 0.8
        theta = np.random.random() * 2 * np.pi
        phi = np.random.random() * np.pi
        
        # Вычисление внутренних координат с использованием сферических координат
        x = r * 16 * np.sin(theta)**3 * np.sin(phi)
        y = r * (13 * np.cos(theta) - 5 * np.cos(2*theta) - 
                2 * np.cos(3*theta) - np.cos(4*theta)) * np.sin(phi)
        z = r * 15 * np.cos(phi)
        points.append((x, y, z))

    # Масштабирование точек
    return scale * np.array(points)

def draw_heart(points, width=80, height=40, time_val=0):
    """
    Отрисовка сердца в терминале с использованием символов, z-буфера и цвета
    
    Параметры:
    - points: массив точек сердца
    - width: ширина терминала
    - height: высота терминала
    - time_val: текущее время для эффектов анимации
    
    Возвращает строку для вывода в терминал
    """
    # Набор символов для имитации глубины
    shading_chars = " .:!*OQ#"
    
    # Нормализация координат для отображения в терминале
    x = (points[:, 0] / np.max(np.abs(points[:, 0])) * (width//2) + width//2).astype(int)
    y = (-points[:, 1] / np.max(np.abs(points[:, 1])) * (height//2) + height//2).astype(int)
    z = points[:, 2]
    
    # Отсечение точек за пределами экрана
    mask = (0 <= x) & (x < width) & (0 <= y) & (y < height)
    x, y, z = x[mask], y[mask], z[mask]
    
    # Создание экрана и z-буфера
    screen = np.full((height, width), ' ', dtype=object)
    z_buffer = np.full((height, width), float('-inf'))
    
    # Отрисовка точек с учетом глубины
    if len(z) >  0:
        z_min, z_max = np.min(z), np.max(z)
        if z_max > z_min:
            # Нормализация глубины
            z_normalized = (z - z_min) / (z_max - z_min)
            # Преобразование глубины в индекс символа
            intensity = (z_normalized * (len(shading_chars) - 1)).astype(int)
            
            # Добавление эффекта мерцания
            brightness = 0.5 + 0.5 * math.sin(time_val * 2 * np.pi)  # Мерцание от 0.0 до 1.0
            
            # Отрисовка точек с учетом z-буфера и интенсивности
            for xi, yi, zi, char_index in zip(x, y, z, intensity):
                if zi > z_buffer[yi, xi]:
                    z_buffer[yi, xi] = zi
                    z_factor = (zi - z_min) / (z_max - z_min)
                    hue = (time_val + z_factor) % 1.0
                    # Применение эффекта мерцания к цвету
                    color_hue = (hue * brightness) % 1.0
                    screen[yi, xi] = get_colored_char(shading_chars[char_index], color_hue)
    
    # Преобразование экрана в строку
    return '\n'.join(''.join(row) for row in screen)

def pulsating_effect(time):
    """
    Создание эффекта пульсации
    
    Параметр:
    - time: текущее время для вычисления пульсации
    
    Возвращает значение масштаба для эффекта пульсации
    """
    return 1 + 0.05 * math.sin(time * 2)  # Пульсация от 1 до 1.05

def main():
    """
    Основная функция программы, отвечающая за инициализацию и выполнение цикла отрисовки
    """
    # Создание начальных точек сердца
    heart_points = create_heart_points(scale=8)
    angle_y = 0  # Начальный угол поворота вокруг оси Y
    
    # Очищаем экран и скрываем курсор
    print('\033[2J')
    print('\033[?25l')
    
    # Создаем очередь для подсчета FPS
    fps_counter = deque(maxlen=60)
    start_time = time.time()  # Запоминаем время начала
    
    try:
        while True:
            # Время начала кадра
            frame_start = time.time()
            current_time = frame_start - start_time
            
            # Масштабируем сердце
            scale = pulsating_effect(current_time)
            scaled_points = heart_points * scale
            
            # Поворачиваем сердце
            rotated_points = rotate_points(scaled_points, angle_y)
            
            # Отрисовываем сердце
            frame = draw_heart(rotated_points, time_val=(current_time * 0.1) % 1.0)
            
            # Подсчитываем FPS
            fps_counter.append(time.time())
            fps = calculate_fps(fps_counter)
            
            # Выводим FPS и сердце
            status_line = f"\033[1mFPS: {fps:.1f} | Press Ctrl+C to exit\033[0m"
            frame_with_status = frame + "\n" + status_line
            
            # Центрируем вывод
            terminal_width = 80
            frame_width = len(frame.split('\n')[0])
            padding = ' ' * ((terminal_width - frame_width) // 2)
            frame_with_status = padding + frame_with_status + padding
            
            # Выводим результат
            sys.stdout.write('\033[H' + frame_with_status)
            sys.stdout.flush()
            
            # Обновляем угол поворота
            angle_y += 0.05  # Увеличение угла поворота
            
            # Ограничиваем FPS
            frame_time = time.time() - frame_start
            if frame_time < 0.033:  # Ограничение до 30 FPS
                time.sleep(0.033 - frame_time)

    except KeyboardInterrupt:
        # Восстанавливаем курсор и очищаем экран
        print('\033[?25h')
        print("\nProgram terminated")

if __name__ == "__main__":
    try:
        main()  # Запуск основной функции программы
    except KeyboardInterrupt:
        print('\033[?25h')
        print("\nProgram terminated")


# Объяснение изменений:

# В функции draw_heart добавлена переменная brightness, которая меняется от 0.0 до 1.0 с помощью функции math.sin.
# Это создает эффект мерцания.

# color_hue рассчитывается с учетом brightness, чтобы цвет символов варьировался в зависимости от значения brightness.