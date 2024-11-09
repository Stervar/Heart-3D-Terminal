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

def create_heart_points(scale=15):
    """
    Генерация точек для создания 3D модели сердца
    
    Args:
        scale (int): Масштаб сердца
    
    Returns:
        np.array: Массив точек сердца
    """
    # Параметрическое уравнение сердца с оптимизацией количества точек
    t = np.linspace(0, 2*np.pi, 1000)
    
    # Формула сердца с использованием тригонометрических функций
    x = 16 * np.sin(t)**3
    y = -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
    z = np.zeros_like(x)

    # Создание слоев сердца с постепенным уменьшением
    for i in range(50):
        factor = 1 - i/50  # Коэффициент уменьшения
        x = np.append(x, factor * 16 * np.sin(t)**3)
        y = np.append(y, factor * -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)))
        z = np.append(z, np.full_like(t, i/3))

    # Возвращаем массив точек с масштабированием
    return scale * np.column_stack((x, y, z))

def rotate_points(points, angle_x, angle_y, angle_z):
    """
    Функция поворота точек вокруг оси Y (упрощенная версия)
    
    Args:
        points (np.array): Массив точек для поворота
        angle_x (float): Угол поворота вокруг оси X (не используется)
        angle_y (float): Угол поворота вокруг оси Y
        angle_z (float): Угол поворота вокруг оси Z (не используется)
    
    Returns:
        np.array: Повернутые точки
    """
    # Матрица поворота вокруг оси Y
    rotation = np.array([
        [cos(angle_y), 0, sin(angle_y)],
        [0, 1, 0],
        [-sin(angle_y), 0, cos(angle_y)]
    ])
    
    # Применение матрицы поворота
    return np.dot(points, rotation.T)

def draw_heart(points, width=80, height=40):
    """
    Отрисовка сердца с использованием символов и z-буфера
    
    Args:
        points (np.array): Точки сердца
        width (int): Ширина экрана
        height (int): Высота экрана
    """
    # Инициализация экрана и z-буфера
    screen = [[' ' for _ in range(width)] for _ in range(height)]
    z_buffer = [[float('-inf') for _ in range(width)] for _ in range(height)]
    
    # Символы для создания эффекта затенения
    shading_chars = " .:!*oe%&#@"
    
    # Извлечение координат точек
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Проекция 3D точек на 2D экран с центрированием
    x = (x / np.max(np.abs(x)) * (width//4)) + width//2
    y = (y / np.max(np.abs(y)) * (height//4)) + height//2
    
    # Отрисовка точек с учетом глубины
    for xi, yi, zi in zip(x, y, z):
        xi, yi = int(xi), int(yi)
        if 0 <= xi < width and 0 <= yi < height:
            if zi > z_buffer[yi][xi]:
                z_buffer[yi][xi] = zi
                # Нормализация интенсивности для выбора символа
                intensity = int((zi - np.min(z)) / (np.max(z) - np.min(z)) * (len(shading_chars) - 1))
                screen[yi][xi] = shading_chars[intensity]
    
    # Вывод экрана с использованием ANSI-escape последовательностей
    print('\033[H' + '\n'.join(''.join(row) for row in screen))

def main():
    """
    Основная функция для запуска анимации сердца
    """
    # Генерация точек сердца
    heart_points = create_heart_points()
    
    # Начальные углы вращения
    angle_x = angle_y = angle_z = 0
    
    # Подготовка терминала
    print('\033[2J')      # Очистка экрана
    print('\033[?25l')    # Скрытие курсора
    
    try:
        while True:
            # Вращение точек сердца
            rotated_points = rotate_points(heart_points, angle_x, angle_y, angle_z)
            
            # Отрисовка сердца
            draw_heart(rotated_points)
            
            # Обновление углов вращения с увеличенной скоростью
            angle_y += 0.3
            angle_x += 0.1
            angle_z += 0.05
            
    except KeyboardInterrupt:
        # Обработка завершения программы
        print('\033[?25h')  # Показать курсор
        print("\nПрограмма завершена")

if __name__ == "__main__":
    try:
        main()  # Запуск основной функции
    except KeyboardInterrupt:
        print('\033[?25h')  # Показать курсор
        print("\nПрограмма завершена")