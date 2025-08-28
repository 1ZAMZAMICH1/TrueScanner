# D:\TrueScanner\analyzer.py
import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def analyze_sheet(image_path, correct_keys, template_info):
    try:
        # 1. ЗАГРУЗКА И ПОДГОТОВКА ИЗОБРАЖЕНИЯ
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        # 2. ПОИСК КОНТУРА БЛАНКА
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        doc_contour = None
        
        if len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    doc_contour = approx
                    break
        
        if doc_contour is None:
            raise ValueError("Не удалось найти контур бланка на изображении.")

        # 3. ВЫРАВНИВАНИЕ ПЕРСПЕКТИВЫ
        points = doc_contour.reshape(4, 2)
        rect = order_points(points)
        (tl, tr, br, bl) = rect
        
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")
            
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(gray, M, (max_width, max_height))

        # 4. АНАЛИЗ КРУЖОЧКОВ
        thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        student_answers = {}
        options_count = len(template_info['options'])
        questions_per_column = int(np.ceil(template_info['questions_count'] / 2))

        for q_index in range(template_info['questions_count']):
            col = 0 if q_index < questions_per_column else 1
            row = q_index if col == 0 else q_index - questions_per_column
            
            darkest_bubble_val = 255
            student_choice = None
            
            for opt_index in range(options_count):
                # Эти проценты нужно будет точно подбирать под твой бланк
                y_start_percent = 0.15 + (row * 0.052)
                x_start_percent = (0.15 if col == 0 else 0.58) + (opt_index * 0.07)
                
                x = int(x_start_percent * max_width)
                y = int(y_start_percent * max_height)
                w = int(0.05 * max_width)
                h = int(0.03 * max_height)

                bubble_roi = thresh[y:y+h, x:x+w]
                if bubble_roi.size == 0: continue

                total_white_pixels = cv2.countNonZero(bubble_roi)
                
                # Рисуем рамки для отладки
                cv2.rectangle(warped, (x, y), (x+w, y+h), (128, 128, 128), 1)

                if total_white_pixels < darkest_bubble_val:
                    darkest_bubble_val = total_white_pixels
                    student_choice = template_info['options'][opt_index]
            
            student_answers[q_index + 1] = student_choice

        # 5. ПРОВЕРКА И ВИЗУАЛИЗАЦИЯ
        correct_count = 0
        incorrect_count = 0
        
        # ... (здесь будет логика подсчета и отрисовки результатов на изображении)

        return warped, f"Правильно: {correct_count}, Неправильно: {incorrect_count}"

    except Exception as e:
        print(f"Ошибка анализа: {e}")
        return None, "Ошибка анализа изображения"