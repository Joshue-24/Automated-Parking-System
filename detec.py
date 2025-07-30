import cv2
from ultralytics import YOLO
import numpy as np
import time
import os
import easyocr
import re

# Inicializar EasyOCR (usando solo inglés para mayor velocidad)
print("Inicializando EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)  # Cambiar a gpu=True si tienes una GPU compatible
print("✅ EasyOCR inicializado")

def preprocess_plate_image(plate_img):
    """Preprocesa la imagen de la placa para mejorar el OCR"""
    # Convertir a escala de grises
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Aumentar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    contrast = clahe.apply(gray)

def clean_plate_text(text):
    """Limpia el texto de la placa detectada"""
    # Eliminar caracteres no alfanuméricos
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    try:
        # Reducir tamaño para procesamiento más rápido
        h, w = image.shape[:2]
        aspect = w / float(h)
        new_w = int(target_height * aspect)
        
        # Redimensionar manteniendo relación de aspecto
        resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Aumentar contraste con CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(gray)
        
        # Aplicar umbral adaptativo
        thresh = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Operación de apertura para eliminar ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return processed
    except Exception as e:
        print(f"Error en preprocesamiento: {e}")
        return None

def read_plate_text(image, reader=None):
    """
    Procesa la imagen de una placa y devuelve el texto detectado.
    Usa EasyOCR para el reconocimiento de caracteres.
    """
    try:
        # Inicializar el lector OCR si no está inicializado
        if reader is None:
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aumentar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Aplicar umbral adaptativo
        thresh = cv2.adaptiveThreshold(enhanced, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Leer texto
        results = reader.readtext(thresh, 
                               detail=0, 
                               paragraph=True, 
                               batch_size=1,
                               decoder='greedy',
                               allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        # Procesar resultados
        if results:
            # Tomar el primer resultado
            text = results[0].upper()
            # Limpiar el texto (mantener solo letras mayúsculas y números)
            clean_text = re.sub(r'[^A-Z0-9]', '', text)
            # Validar formato básico
            if 4 <= len(clean_text) <= 10 and any(c.isdigit() for c in clean_text) and any(c.isalpha() for c in clean_text):
                return clean_text
        
        return "No detectado"
        
    except Exception as e:
        print(f"Error en OCR: {e}")
        return "Error"

# Configuración
TARGET_WIDTH = 640  # Ancho objetivo para el procesamiento
CONFIDENCE_THRESHOLD = 0.4  # Umbral de confianza mínimo para vehículos
FPS_UPDATE_INTERVAL = 10  # Actualizar FPS cada X frames

# Clase de interés: solo coches (class_id = 2 en COCO dataset)
CAR_CLASS = [2]

print("Cargando modelos...")
# Cargar modelo de vehículos
vehicle_model = YOLO('yolov8n.pt')  # Modelo YOLOv8 nano (más rápido)
print("✅ Modelo de vehículos cargado")

# Cargar modelo de placas
if os.path.exists('placa.pt'):
    try:
        plate_model = YOLO('placa.pt')
        print("✅ Modelo de placas cargado correctamente")
    except Exception as e:
        print(f"❌ Error al cargar el modelo de placas: {e}")
        plate_model = None
else:
    print("⚠️ No se encontró el archivo 'placa.pt'")
    plate_model = None

# Configurar OpenCV para mejor rendimiento
cv2.setNumThreads(1)

def detect_vehicles():
    # Inicializar variables para el seguimiento de placas
    plate_texts = {}  # Diccionario para mantener el texto de las placas por ID de vehículo
    frame_count = 0
    ocr_update_interval = 3  # Actualizar OCR cada 3 frames
    reader = None  # Inicializar el lector OCR una sola vez
    
    # Inicializar la cámara web
    print("Inicializando la cámara...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_WIDTH * 9 // 16)  # Relación de aspecto 16:9
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Verificar que la cámara se abrió correctamente
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return

    print("Cámara lista. Presiona 'q' para salir")

    # Variables para el cálculo de FPS
    frame_count = 0
    fps = 0
    start_time = time.time()
    prev_frame_time = time.time()

    # Bucle principal para captura y detección
    while True:
        # Calcular FPS
        current_time = time.time()
        frame_count += 1
        
        # Actualizar FPS cada FPS_UPDATE_INTERVAL frames
        if frame_count % FPS_UPDATE_INTERVAL == 0:
            fps = FPS_UPDATE_INTERVAL / (current_time - start_time)
            start_time = current_time
        
        # Leer un frame de la cámara
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame")
            break
        
        # Redimensionar frame para procesamiento más rápido
        height, width = frame.shape[:2]
        if width > TARGET_WIDTH:
            scale = TARGET_WIDTH / width
            frame = cv2.resize(frame, (TARGET_WIDTH, int(height * scale)), 
                             interpolation=cv2.INTER_AREA)
        
        # Realizar la detección de vehículos
        results = vehicle_model(frame, 
                              verbose=False, 
                              conf=CONFIDENCE_THRESHOLD,
                              classes=[2])  # Solo detectar coches
        
        # Mostrar FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Procesar los resultados
        for result in results:
            # Obtener las cajas delimitadoras
            boxes = result.boxes
            
            # Dibujar cada caja detectada
            for box in boxes:
                # Obtener coordenadas (usar GPU si está disponible)
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Dibujar rectángulo del vehículo
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Mostrar etiqueta y confianza del vehículo
                vehicle_label = f"{vehicle_model.names[class_id]} {conf:.1f}"
                cv2.putText(frame, vehicle_label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Buscar placas en la región del vehículo
                if plate_model is not None:  # Procesar en cada frame
                    try:
                        # Definir región de interés (parte inferior del vehículo)
                        roi_height = int((y2 - y1) * 0.4)  # 40% de la altura del vehículo para mayor cobertura
                        roi_y1 = max(0, y1 + int((y2 - y1) * 0.6))  # Comenzar desde el 60% de la altura
                        roi_y2 = min(roi_y1 + roi_height, frame.shape[0] - 1)
                        roi_x1 = max(0, x1 - 10)  # Añadir un pequeño margen horizontal
                        roi_x2 = min(x2 + 10, frame.shape[1] - 1)
                        
                        # Asegurar dimensiones mínimas
                        if roi_y2 > roi_y1 + 20 and roi_x2 > roi_x1 + 20:
                            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                            
                            # Detectar placas en la ROI con un umbral de confianza más bajo para mayor sensibilidad
                            plate_results = plate_model(roi, verbose=False, conf=0.25)
                            
                            # Generar un ID único para este vehículo basado en su posición
                            vehicle_id = f"{x1}_{y1}_{x2}_{y2}"
                            
                            if len(plate_results) > 0 and len(plate_results[0].boxes) > 0:
                                # Tomar solo la placa con mayor confianza
                                best_box = max(plate_results[0].boxes, key=lambda x: float(x.conf[0]))
                                
                                # Obtener coordenadas de la placa en la ROI
                                px1_roi, py1_roi, px2_roi, py2_roi = map(int, best_box.xyxy[0].cpu().numpy())
                                conf = float(best_box.conf[0])
                                
                                # Ajustar coordenadas al frame original
                                px1 = px1_roi + roi_x1
                                px2 = px2_roi + roi_x1
                                py1 = py1_roi + roi_y1
                                py2 = py2_roi + roi_y1
                                
                                # Asegurar que las coordenadas estén dentro de los límites del frame
                                px1, py1 = max(0, px1), max(0, py1)
                                px2, py2 = min(frame.shape[1] - 1, px2), min(frame.shape[0] - 1, py2)
                                
                                if px2 > px1 and py2 > py1:  # Verificar dimensiones válidas
                                    # Extraer la región de la placa
                                    plate_roi = frame[py1:py2, px1:px2]
                                    
                                    # Leer texto de la placa con actualización periódica
                                    if plate_roi.size > 0 and (frame_count % ocr_update_interval == 0 or vehicle_id not in plate_texts):
                                        # Inicializar el lector OCR en el primer uso
                                        if reader is None:
                                            reader = easyocr.Reader(['en'], gpu=False, verbose=False)

                                        try:
                                            # Asegurar tamaño mínimo para el OCR
                                            h, w = plate_roi.shape[:2]
                                            print(f"Tamaño de ROI de placa: {w}x{h}")
                                            if h > 20 and w > 60:  # Tamaño mínimo razonable para una placa
                                                print("Procesando ROI de placa...")
                                                plate_text = read_plate_text(plate_roi, reader)
                                                print(f"Texto detectado: {plate_text}")

                                                if plate_text not in ["No detectado", "Error"]:
                                                    # Solo actualizar si el nuevo texto tiene más caracteres o es más confiable
                                                    if (vehicle_id not in plate_texts or
                                                        len(plate_text) > len(plate_texts[vehicle_id])):
                                                        plate_texts[vehicle_id] = plate_text
                                                        print(f"Placa actualizada para vehículo {vehicle_id}: {plate_text}")
                                            else:
                                                print(f"ROI demasiado pequeña: {w}x{h}")
                                        except Exception as e:
                                            print(f"Error procesando placa: {e}")

                                    # Actualizar el contador de frames
                                    frame_count = (frame_count + 1) % (ocr_update_interval * 2)

                                    # Mostrar rectángulo de la placa
                                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
                                    
                                    # Mostrar texto de la placa si está disponible
                                    current_plate_text = plate_texts.get(vehicle_id, "")
                                    
                                    # Solo mostrar si hay texto válido
                                    if current_plate_text not in ["No detectado", "Error", ""]:
                                        # Texto sobre la placa
                                        cv2.putText(frame, current_plate_text, 
                                                  (px1, py1 - 10),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                                  0.7, (0, 255, 0), 2)
                                        
                                        # Texto en la parte superior
                                        cv2.putText(frame, f"Placa: {current_plate_text}", 
                                                  (20, 40), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                                  0.8, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Error en detección de placa: {e}")
        
        # Mostrar el frame con las detecciones
        cv2.imshow('Detección de Vehículos', frame)
        
        # Calcular tiempo de procesamiento
        processing_time = time.time() - current_time
        
        # Esperar para mantener una tasa de cuadros constante
        wait_time = max(1, int((1/30 - processing_time) * 1000))  # Objetivo: 30 FPS
        
        # Salir con 'q' o ESC
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q') or key == 27:  # 'q' o ESC
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("\n¡Programa finalizado!")

# Llamar a la función principal
if __name__ == "__main__":
    detect_vehicles()