import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import threading
import os
import matplotlib.pyplot as plt
from emotion_analyzer import EmotionAnalyzer

class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Zihin Durumu Analizi')
        self.root.geometry('900x700')
        self.analyzer = EmotionAnalyzer()
        self.cap = None
        self.running = False
        self.frame = None
        self.update_job = None
        self.video_thread = None
        self.video_running = False

        # Arayüz elemanları
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        self.result_text = tk.StringVar()
        self.result_label = tk.Label(root, textvariable=self.result_text, font=('Arial', 14))
        self.result_label.pack(pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        self.start_btn = ttk.Button(btn_frame, text='Analizi Başlat', command=self.start_analysis)
        self.start_btn.grid(row=0, column=0, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text='Durdur', command=self.stop_analysis, state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=5)
        self.graph_btn = ttk.Button(btn_frame, text='Grafiği Göster', command=self.show_graph)
        self.graph_btn.grid(row=0, column=2, padx=5)
        self.video_btn = ttk.Button(btn_frame, text='Video ile Analiz', command=self.analyze_video)
        self.video_btn.grid(row=0, column=3, padx=5)

    def start_analysis(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.update_frame()

    def stop_analysis(self):
        self.running = False
        self.video_running = False  # Video analizini de durdur
        # Butonları hemen pasif yapma, thread bitince ana thread'de güncellenecek
        if self.cap:
            self.cap.release()
        # Sadece görüntüyü temizle
        self.video_label.config(image=None)
        self.result_text.set('')
        self.analyzer.visualize_results()
        self.analyzer.get_statistics()

    def update_frame(self):
        if self.running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                analyzed = self.analyzer.analyze_frame(frame.copy())
                # Sonuçları ekrana yaz
                if self.analyzer.emotion_data:
                    last = self.analyzer.emotion_data[-1]
                    emotion_str = ', '.join([f"{k}: {v:.1f}" for k, v in last.items()])
                else:
                    emotion_str = ''
                if hasattr(self.analyzer.body_analyzer, 'pose_data') and self.analyzer.body_analyzer.pose_data:
                    last_pose = self.analyzer.body_analyzer.pose_data[-1]
                    pose_str = f"Duruş: {last_pose['posture']}, Hareket: {last_pose['movement']}"
                else:
                    pose_str = ''
                self.result_text.set(f"Duygular: {emotion_str}\nVücut Dili: {pose_str}")
                # Görüntüyü göster
                img = cv2.cvtColor(analyzed, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = img.resize((640, 480))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
            self.update_job = self.root.after(30, self.update_frame)
        else:
            if self.cap:
                self.cap.release()

    def analyze_video(self):
        video_path = filedialog.askopenfilename(
            title='Video Seç',
            filetypes=[('Video Dosyaları', '*.mp4 *.avi *.mov *.mkv')]
        )
        if not video_path:
            return
        self.result_text.set('Video analizi başlatıldı...')
        self.video_running = True
        self.stop_btn.config(state='normal')
        self.video_thread = threading.Thread(target=self._analyze_video_thread, args=(video_path,))
        self.video_thread.start()

    def _analyze_video_thread(self, video_path):
        cap = cv2.VideoCapture(video_path)
        self.analyzer.emotion_data = []
        self.analyzer.timestamps = []
        self.analyzer.body_analyzer.pose_data = []
        self.analyzer.body_analyzer.timestamps = []
        frame_count = 0
        while self.video_running:
            ret, frame = cap.read()
            if not ret:
                break
            analyzed = self.analyzer.analyze_frame(frame.copy())
            # Sonuçları ekrana yaz
            if self.analyzer.emotion_data:
                last = self.analyzer.emotion_data[-1]
                emotion_str = ', '.join([f"{k}: {v:.1f}" for k, v in last.items()])
            else:
                emotion_str = ''
            if hasattr(self.analyzer.body_analyzer, 'pose_data') and self.analyzer.body_analyzer.pose_data:
                last_pose = self.analyzer.body_analyzer.pose_data[-1]
                pose_str = f"Duruş: {last_pose['posture']}, Hareket: {last_pose['movement']}"
            else:
                pose_str = ''
            self.result_text.set(f"Duygular: {emotion_str}\nVücut Dili: {pose_str}\nKare: {frame_count}")
            # Görüntüyü göster
            img = cv2.cvtColor(analyzed, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
            frame_count += 1
            self.root.update()
        cap.release()
        # Thread bittiğinde ana thread'de arayüzü güncelle
        self.root.after(0, self._on_video_analysis_end)

    def _on_video_analysis_end(self):
        self.result_text.set('Video analizi tamamlandı.')
        self.analyzer.visualize_results()
        self.analyzer.get_statistics()
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')

    def show_graph(self):
        path = 'emotion_analysis_simple.png'
        if not os.path.exists(path):
            messagebox.showinfo('Bilgi', 'Henüz analiz grafiği oluşturulmadı.')
            return
        img = Image.open(path)
        img.show()

if __name__ == '__main__':
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop() 