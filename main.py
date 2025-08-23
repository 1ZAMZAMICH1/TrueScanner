# D:\TrueScanner\main.py
import cv2
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.core.image import Image as CoreImage
from kivy.graphics.texture import Texture
import io

# --- СЕРДЦЕ ПРИЛОЖЕНИЯ: ДВИЖОК OPENCV ---
def find_anchors_and_draw(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, "Не удалось прочитать изображение"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_anchors = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > 100 and area < 5000:
                cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
                found_anchors += 1

    print(f"Найдено якорей: {found_anchors}")
    return img, f"Найдено якорей: {found_anchors}"

# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ НА KIVY ---
class ScannerApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.img_widget = Image(source='placeholder.png')
        self.result_label = Label(text="Нажмите 'Анализ'", size_hint_y=0.1)
        btn = Button(text="Анализ (Используем тестовое фото)", size_hint_y=0.1)
        btn.bind(on_press=self.analyze_image)
        self.layout.add_widget(self.img_widget)
        self.layout.add_widget(self.result_label)
        self.layout.add_widget(btn)
        return self.layout

    def analyze_image(self, instance):
        processed_img, result_text = find_anchors_and_draw('test_image.jpg')
        if processed_img is not None:
            buf = cv2.imencode('.jpg', processed_img)[1].tobytes()
            texture = CoreImage(io.BytesIO(buf), ext='jpg').texture
            self.img_widget.texture = texture
        self.result_label.text = result_text

if __name__ == '__main__':
    ScannerApp().run()