# D:\TrueScanner\main.py
import os
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.core.image import Image as CoreImage
from kivy.graphics.texture import Texture
import cv2
import io
# analyzer.py мы пока не используем, чтобы упростить. Вся логика будет здесь.

# Загружаем наш файл с дизайном
Builder.load_file('scanner.kv')

class MainScreen(Screen):
    pass

class ResultScreen(Screen):
    def show_result(self, processed_img, result_text):
        if processed_img is not None:
            buf = cv2.imencode('.jpg', processed_img)[1].tobytes()
            texture = CoreImage(io.BytesIO(buf), ext='jpg').texture
            self.ids.result_image.texture = texture
        self.ids.result_label.text = result_text

class ScannerScreen(Screen):
    def capture_and_analyze(self):
        camera = self.ids['camera']
        # Делаем "снимок" не на диск, а в память
        camera.export_to_png("temp_capture.png") 
        print("Снимок сделан во временный файл: temp_capture.png")
        self.analyze("temp_capture.png")

    def analyze(self, image_path):
        # --- НАСТОЯЩИЙ АНАЛИЗАТОР OPENCV ---
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Не удалось прочитать временный файл")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 150)
            
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            found_anchors = 0
            if len(contours) > 0:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                for c in contours:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                    if len(approx) == 4:
                        area = cv2.contourArea(approx)
                        if area > 500: # Ищем большие контуры
                            cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
                            found_anchors += 1
            
            result_text = f"Найдено якорей: {found_anchors}"
            
            # Переключаемся на экран результатов и показываем картинку
            result_screen = self.manager.get_screen('results')
            result_screen.show_result(img, result_text)
            self.manager.current = 'results'

        except Exception as e:
            print(f"Ошибка анализа: {e}")
            result_screen = self.manager.get_screen('results')
            result_screen.show_result(None, f"Ошибка: {e}")
            self.manager.current = 'results'


class ScannerApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(ScannerScreen(name='scanner'))
        sm.add_widget(ResultScreen(name='results'))
        return sm

if __name__ == '__main__':
    ScannerApp().run()