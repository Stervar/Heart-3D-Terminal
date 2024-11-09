import numpy as np  # Библиотека для работы с массивами и математическими вычислениями
import time  # Для работы со временем и задержками
import sys  # Для взаимодействия с системой (вывод в терминал)
import math  # Для математических функций (sin, cos)
from collections import deque  # Двусторонняя очередь для подсчета FPS
import colorsys  # Для работы с цветовыми пространствами (HSV в RGB)


def rotate_points(points, angle_y):
    # Создаем матрицу поворота вокруг оси Y
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],     # Первая строка матрицы
        [0, 1, 0],                                 # Вторая строка матрицы
        [-np.sin(angle_y), 0, np.cos(angle_y)]     # Третья строка матрицы
    ])
    # Умножаем точки на транспонированную матрицу поворота
    return np.dot(points, Ry.T)


def calculate_fps(fps_counter):
    if len(fps_counter) < 2:  # Проверка наличия минимум двух временных меток
        return 0.0
    
    
    # Вычисляем разницу времени между первой и последней меткой
    time_diff = fps_counter[-1] - fps_counter[0]
    

    if time_diff <= 0:  # Защита от деления на ноль
        return 0.0


    # Возвращаем количество кадров / время = FPS
    return len(fps_counter) / time_diff


def get_colored_char(char, hue, saturation=1.0, value=1.0):
    # Ограничиваем оттенок в диапазоне [0, 1]
    hue = max(0.0, min(1.0, hue))
    

    # Преобразуем HSV в RGB, масштабируем до 255
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value)]
    

    # Формируем ANSI-последовательность для цветного символа
    return f"\033[38;2;{r};{g};{b}m{char}\033[0m"


def create_heart_points(scale=5, num_points=1000, num_layers=30):
    # Создаем параметрическую кривую
    t = np.linspace(0, 2*np.pi, num_points)
    

    # Параметрические уравнения сердца
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    z = np.zeros_like(x)

    points = []


    # Создаем слои для объема
    for i in range(num_layers):
        factor = 1 - i/num_layers
        layer_x = factor * x
        layer_y = factor * y
        layer_z = np.full_like(x, -i/2)
        points.extend(zip(layer_x, layer_y, layer_z))


    # Добавляем внутренние точки
    for _ in range(num_points // 2):
        r = np.random.random() * 0.8
        theta = np.random.random() * 2 * np.pi
        phi = np.random.random() * np.pi
        

        # Вычисляем координаты внутренних точек
        x = r * 16 * np.sin(theta)**3 * np.sin(phi)
        y = r * (13 * np.cos(theta) - 5 * np.cos(2*theta) - 
                2 * np.cos(3*theta) - np.cos(4*theta)) * np.sin(phi)
        z = r * 15 * np.cos(phi)
        points.append((x, y, z))


    # Масштабируем все точки
    return scale * np.array(points)



def draw_heart(points, width=80, height=40, time_val=0):
    # Символы для отображения глубины
    shading_chars = " .:!*OQ#"
    
    # Проецируем 3D координаты на 2D экран
    x = (points[:, 0] / np.max(np.abs(points[:, 0])) * (width//2) + width//2).astype(int)


    # Изменяем знак для y-координаты, чтобы перевернуть сердце
    y = (-points[:, 1] / np.max(np.abs(points[:, 1])) * (height//2) + height//2).astype(int)

    z = points[:, 2]
    

    # Отсекаем точки за пределами экрана
    mask = (0 <= x) & (x < width) & (0 <= y) & (y < height)
    x, y, z = x[mask], y[mask], z[mask]
    

    # Создаем экран и z-буфер
    screen = np.full((height, width), ' ', dtype=object)
    z_buffer = np.full((height, width), float('-inf'))
    

    # Отрисовка точек с учетом глубины
    if len(z) > 0:
        z_min, z_max = np.min(z), np.max(z)
        if z_max > z_min:
            z_normalized = (z - z_min) / (z_max - z_min)
            intensity = (z_normalized * (len(shading_chars) - 1)).astype(int)
            
            for xi, yi, zi, char_index in zip(x, y, z, intensity):
                if zi > z_buffer[yi, xi]:
                    z_buffer[yi, xi] = zi
                    z_factor = (zi - z_min) / (z_max - z_min)
                    hue = (time_val + z_factor) % 1.0
                    screen[yi, xi] = get_colored_char(shading_chars[char_index], hue)
    
    return '\n'.join(''.join(row) for row in screen)



def pulsating_effect(time):
    # Создает синусоидальное изменение размера
    return 1 + 0.05 * math.sin(time * 2)



def main():
    # Создаем начальные точки сердца
    heart_points = create_heart_points(scale=8)
    angle_y = 0
    

    # Очищаем экран и скрываем курсор
    print('\033[2J')
    print('\033[?25l')
    

    # Создаем очередь для подсчета FPS
    fps_counter = deque(maxlen=60)
    start_time = time.time()
    
    
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
            angle_y += 0.05
            

            # Ограничиваем FPS
            frame_time = time.time() - frame_start
            if frame_time < 0.033:
                time.sleep(0.033 - frame_time)


    except KeyboardInterrupt:
        # Восстанавливаем курсор и очищаем экран
        print('\033[?25h')
        print("\nProgram terminated")
        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\033[?25h')
        print("\nProgram terminated")
        
        
        
        
        
        
        
        
        
        
        
        
        
# Импорты:

# import numpy as np
# import time
# import sys
# import math
# from collections import deque
# import colorsys

# numpy (np) - библиотека для работы с массивами и математическими операциями

# time - для работы со временем

# sys - для взаимодействия с системой

# math - для математических функций

# deque - структура данных "двусторонняя очередь"

# colorsys - для работы с цветовыми пространствами





# Функция rotate_points:


# def rotate_points(points, angle_y):
#     Ry = np.array([
#         [np.cos(angle_y), 0, np.sin(angle_y)],
#         [0, 1, 0],
#         [-np.sin(angle_y), 0, np.cos(angle_y)]
#     ])
#     return np.dot(points, Ry.T)
# Эта функция выполняет поворот точек вокруг оси Y:

# Создает матрицу поворота Ry

# Умножает точки на транспонированную матрицу поворота

# Возвращает повернутые точки





# Функция calculate_fps:


# def calculate_fps(fps_counter):
#     if len(fps_counter) < 2:
#         return 0.0
#     time_diff = fps_counter[-1] - fps_counter[0]
#     if time_diff <= 0:
#         return 0.0
#     return len(fps_counter) / time_diff
# Рассчитывает количество кадров в секунду:

# Проверяет наличие минимум двух временных меток

# Вычисляет разницу времени между первой и последней меткой

# Возвращает FPS как количество кадров, деленное на время






# Функция get_colored_char:


# def get_colored_char(char, hue, saturation=1.0, value=1.0):
#     hue = max(0.0, min(1.0, hue))
#     r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value)]
#     return f"\033[38;2;{r};{g};{b}m{char}\033[0m"

# Создает цветной символ для терминала:

# Преобразует HSV в RGB

# Форматирует ANSI-последовательность для цветного вывода

# Возвращает цветной символ







# Функция create_heart_points:


# def create_heart_points(scale=5, num_points=1000, num_layers=30):
    
    
# Создает точки, формирующие 3D сердце:
    
# Использует параметрические уравнения для создания формы сердца

# Создает несколько слоев для объема

# Добавляет случайные внутренние точки

# Масштабирует все точки









# Функция draw_heart:


# def draw_heart(points, width=80, height=40, time_val=0):
   
    
# Отрисовывает сердце в терминале:

# Проецирует 3D точки на 2D экран

# Использует z-буфер для правильного отображения глубины

# Применяет затенение и цвета

# Возвращает строку для вывода в терминал








# Функция pulsating_effect:


# def pulsating_effect(time):
#     return 1 + 0.05 * math.sin(time * 2)

# Создает эффект пульсации:

# Использует синусоиду для создания периодического изменения размера







# Функция main:

# def main():
# Основной цикл программы:

# Инициализирует точки сердца

# Запускает бесконечный цикл анимации

# Обновляет положение и размер сердца

# Выводит результат в терминал

# Контролирует FPS







# Запуск программы:

# if __name__ == "__main__":
# Точка входа в программу с обработкой прерывания Ctrl+C










# Параметрические уравнения - это способ описания кривых или поверхностей,
# где координаты точек выражаются как функции от одного или нескольких параметров.


# Что такое параметрические уравнения?
# Параметрические уравнения описывают координаты точки как функции от некоторого параметра (обычно обозначается t). В общем виде они выглядят так:


# x = f(t)
# y = g(t)
# z = h(t)  // для 3D



# В данном коде используются следующие параметрические уравнения для сердца:

# t = np.linspace(0, 2*np.pi, num_points)  # параметр t от 0 до 2π

# x = 16 * np.sin(t)**3                    # уравнение для x
# y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)  # уравнение для y
# Разбор уравнений:


#1) Уравнение для x:

# 16 * np.sin(t)**3
# Куб синуса создает характерную форму верхних долей сердца
# Коэффициент 16 отвечает за масштаб


#2) Уравнение для y:

# 13 * np.cos(t) - основная форма
# -5 * np.cos(2*t) - первая гармоника
# -2 * np.cos(3*t) - вторая гармоника
# -np.cos(4*t) - третья гармоника
# Сумма этих косинусов создает характерный изгиб внизу сердца




# Как это работает:




#1) При t = 0:

# x = 0
# y принимает максимальное значение



#2) При t = π:

# x = 0
# y принимает минимальное значение



#3) При промежуточных значениях t:

# Точки описывают контур сердца





# Преимущества параметрических уравнений:

# Легко управлять формой через параметр t
# Можно создавать сложные формы
# Удобно для анимации
# Хорошо подходят для компьютерной графики




# В коде эти уравнения используются для:

# def create_heart_points(scale=5, num_points=1000, num_layers=30):
#     t = np.linspace(0, 2*np.pi, num_points)
    
#     # Базовые параметрические уравнения
#     x = 16 * np.sin(t)**3
#     y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
#     z = np.zeros_like(x)  # начальная z-координата



# Затем эти базовые точки используются для создания 3D объекта путем:

# Создания нескольких слоев

# Добавления внутренних точек

# Масштабирования

# Это позволяет создать объемное анимированное сердце в терминале.

