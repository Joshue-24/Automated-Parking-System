import cv2
from ultralytics import YOLO

# Inicializar modelos
model_vehicle = YOLO('yolov8n.pt')  # Modelo para vehículos
model_plate = YOLO('placa.pt')      # Modelo para placas

# Iniciar cámara
cap = cv2.VideoCapture(0)

while True:
    # Leer frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detección de vehículos
    vehicles = model_vehicle(frame, verbose=False)
    
    # Procesar cada vehículo detectado
    for result in vehicles:
        for box in result.boxes:
            if int(box.cls) in [2, 5, 7]:  # Coche, autobús, camión
                # Coordenadas del vehículo
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
                            
                            # Convertir a coordenadas absolutas
                            px1 += x1
                            py1 += y1
                            px2 += x1
                            py2 += y1
                            
                            # Dibujar rectángulo de la placa
                            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
    
    # Mostrar frame
    cv2.imshow('Detección de Vehículos y Placas', frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()