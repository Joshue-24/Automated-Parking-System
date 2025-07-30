import cv2
import pytesseract
from ultralytics import YOLO
import numpy as np

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
model_vehicle = YOLO('yolov8n.pt')  
model_plate = YOLO('placa.pt')     
config_tesseract = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detección de vehículos
    vehicles = model_vehicle(frame, verbose=False)
    
   
    for result in vehicles:
        for box in result.boxes:
            if int(box.cls) in [2, 5, 7]: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Dibujar rectángulo del vehículo
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Obtener región del vehículo
                vehicle_roi = frame[y1:y2, x1:x2]
                
                # Detectar placas solo dentro del vehículo
                if vehicle_roi.size > 0:
                    plates = model_plate(vehicle_roi, verbose=False)
                    
                    # Procesar placas detectadas
                    for plate in plates:
                        for p_box in plate.boxes:
                            # Coordenadas relativas al vehículo
                            px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                            
                         
                            px1 = max(0, px1)
                            py1 = max(0, py1)
                            px2 = min(vehicle_roi.shape[1], px2)
                            py2 = min(vehicle_roi.shape[0], py2)
                            plate_roi = vehicle_roi[py1:py2, px1:px2]
                            
                           
                            if plate_roi.size > 0:
                                
                                gray_plate = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                                thresh = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                                
                              
                                plate_text = pytesseract.image_to_string(
                                    thresh, 
                                    config=config_tesseract,
                                    lang='eng'
                                ).strip()
                                
                                
                                plate_text = ''.join(c for c in plate_text if c.isalnum()).upper()
                                
                                
                                px1_abs = px1 + x1
                                py1_abs = py1 + y1
                                px2_abs = px2 + x1
                                py2_abs = py2 + y1
                            
                                cv2.rectangle(frame, (px1_abs, py1_abs), (px2_abs, py2_abs), (0, 0, 255), 2)
                                
                                # Mostrar texto reconocido
                                if plate_text:
                                    # Fondo para el texto
                                    (text_w, text_h), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                    cv2.rectangle(frame, 
                                                (px1_abs, py1_abs - 30), 
                                                (px1_abs + text_w, py1_abs), 
                                                (0, 0, 0), -1)
                                    # Texto
                                    cv2.putText(frame, plate_text, (px1_abs, py1_abs - 10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Mostrar frame
    cv2.imshow('Detección de Vehículos y Placas', frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()