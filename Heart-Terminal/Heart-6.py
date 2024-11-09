# Импорт необходимых библиотек для создания 3D сердца в терминале
import numpy as np      # Математические операции и работа с массивами
import time             # Работа со временем и задержками
import os               # Работа с операционной системой
import sys              # Системные операции

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
    # Параметрическое уравнение сердца
    t = np.linspace(0, 2*np.pi, 500)  # Количество точек для формы сердца
    x = 16 * np.sin(t)**3
    y = -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
    z = np.zeros_like(x)

    # Создание слоев сердца с постепенным уменьшением
    for i in range(25):  # Количество слоев
        factor = 1 - i/25  # Коэффициент уменьшения
        x = np.append(x, factor * 16 * np.sin(t)**3)
        y = np.append(y, factor * -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)))
        z = np.append(z, np.full_like(t, i/3))

    # Возвращаем массив точек с масштабированием
    return scale * np.column_stack((x, y, z))

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

def draw_heart(points, width=80, height=40):
    """
    Отрисовка сердца с использованием символов и z-буфера
    
    Args:
        points (np.array): Точки сердца
        width (int): Ширина экрана
        height (int): Высота экрана
    """
    # Инициализация экрана и z-буфера
    screen = np.full((height, width), ' ', dtype=str)
    z_buffer = np.full((height, width), float('-inf'))
    
    # Символы для создания эффекта затенения
    shading_chars = " .:!*oe%&#@"
    
    # Извлечение координат точек
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Проекция 3D точек на 2D экран с центрированием
    x = (x / np.max(np.abs(x)) * (width//4) + width//2).astype(int)
    y = (y / np.max(np.abs(y)) * (height//4) + height//2).astype(int)
    
    # Фильтрация точек в пределах экрана
    mask = (0 <= x) & (x < width) & (0 <= y) & (y < height)
    x, y, z = x[mask], y[mask], z[mask]
    
    # Нормализация глубины для выбора символов
    intensity = ((z - np.min(z)) / (np.max(z) - np.min(z)) * (len(shading_chars) - 1)).astype(int)
    
    # Отрисовка точек с учетом глубины
    for xi, yi, zi, char_index in zip(x, y, z, intensity):
        if zi > z_buffer[yi, xi]:
            z_buffer[yi, xi] = zi
            screen[yi, xi] = shading_chars[char_index]
    
    # Вывод экрана с использованием ANSI-escape последовательностей
    sys.stdout.write('\033[H' + '\n'.join(''.join(row) for row in screen))
    sys.stdout.flush()

def main():
    """
    Основная функция для запуска анимации сердца
    """
    heart_points = create_heart_points()  # Генерация точек сердца
    angle_x = angle_y = angle_z = 0  # Начальные углы вращения
    
    print('\033[2J')  # Очистка экрана
    print('\033[?25l')  # Скрытие курсора
    
    try:
        while True:
            start_time = time.time()  # Время начала кадра
            
            # Вращение точек сердца
            rotated_points = rotate_points(heart_points, angle_x, angle_y, angle_z)
            draw_heart(rotated_points)  # Отрисовка сердца
            
            # Обновление углов вращения
            angle_y += 0.2
            angle_x += 0.1
            angle_z += 0.05
            
            # Регулировка частоты кадров
            elapsed_time = time.time() - start_time
            if elapsed_time < 0.03:  # Пытаемся достичь примерно 30 FPS
                time.sleep(0.03 - elapsed_time)
            
    except KeyboardInterrupt:
        print('\033[?25h')  # Показать курсор
        print("\nПрограмма завершена")  # Завершение программы

if __name__ == "__main__":
    try:
        main()  # Запуск основной функции
    except KeyboardInterrupt:
        print('\033[?25h')  # Показать курсор
        print("\nПрограмма заверш ена")  # Завершение программы