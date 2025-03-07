# OpenCentinel

OpenCentinel es un sistema de vigilancia basado en visión por computadora que detecta movimiento y rostros en tiempo real utilizando una cámara. Desarrollado por **csharpkid**, este proyecto utiliza Python y OpenCV para el procesamiento de video, Flask para la interfaz web, y ofrece notificaciones a través de webhooks.

## Características principales

- **Detección de movimiento**: Identifica objetos en movimiento en el feed de video utilizando el algoritmo MOG2.
- **Detección de rostros**: Detecta rostros humanos cuando se identifica movimiento, utilizando clasificadores Haar Cascade.
- **Interfaz web**: Vista en vivo del feed de video y panel de notificaciones en tiempo real.
- **Notificaciones**: Envía alertas con imágenes a través de webhooks configurables.
- **Historial**: Muestra un registro de eventos con imágenes de regiones de interés (ROIs) y capturas completas.
- **Soporte multiplataforma**: Compatible con cámaras en Windows y Linux.

## Requisitos previos

- Python 3.7 o superior
- Una cámara conectada (webcam o cámara IP)
- Tesseract-OCR instalado (para Windows: especificar la ruta en el código)

### Dependencias

Las dependencias están listadas en `requirements.txt`:

opencv-python
numpy
Flask
requests


Instálalas con:

```bash
pip install -r requirements.txt
```

Instalación de Tesseract-OCR (Windows)
Descarga e instala Tesseract desde su página oficial.
https://github.com/tesseract-ocr/tesseract
Asegúrate de que la ruta en el código apunte a tu instalación:
```bash
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```
# Variables de entorno necesarias

Se requiere la variable de entorno `OPENAI_API_KEY` (https://platform.openai.com/api-keys)

# Instalación
1. Clona el repositorio:
```bash
git clone <URL_DEL_REPOSITORIO>
cd OpenCentinel
```

2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

3. Asegúrate de que tu cámara esté conectada y configurada.

# Uso
1. Ejecuta la aplicación:
```bash
python main.py
```

2. Abre tu navegador y visita:


```bash
http://localhost:5000
```

3. La interfaz mostrará:
- Un feed de video en vivo.
- Un panel de notificaciones con eventos detectados.
- Opciones para configurar un webhook.

# Estructura del proyecto
```bash
OpenCentinel/
├── main.py              # Código principal de la aplicación
├── requirements.txt     # Dependencias del proyecto
├── templates/
│   └── index.html       # Plantilla HTML para la interfaz web
└── README.md            # Este archivo
```
# Descripción del código
* main.py:
   *Clase MotionDetector: Maneja la captura de video, detección de movimiento y rostros, y generación de notificaciones.
   *Configuración del servidor Flask para el streaming de video y la API de notificaciones.
   *Soporte para webhooks y almacenamiento en búfer de notificaciones recientes.
* templates/index.html:
   *Interfaz de usuario responsiva con un diseño moderno.
   *Incluye un feed de video, panel de notificaciones, carrusel de rostros y configurador de webhook.
   *Soporta temas claro/oscuro y es compatible con dispositivos móviles.

# Personalización
* Fuente de video: Modifica self.source en MotionDetector.__init__() para usar una cámara diferente (0 para la predeterminada, o una URL para cámaras IP).
* Umbral de detección: Ajusta varThreshold en createBackgroundSubtractorMOG2 o el área mínima en cv2.contourArea para sensibilidad al movimiento.
* Resolución: Cambia los valores en cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) y cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480).

# Notas técnicas
* El sistema utiliza OpenCV con backends específicos por plataforma (CAP_DSHOW para Windows, CAP_V4L2 para Linux).
* Las imágenes se codifican en base64 para su transmisión a través de webhooks y almacenamiento en notificaciones.
* La interfaz web actualiza las notificaciones cada 2 segundos y los tiempos relativos cada minuto.

# Contribuciones 
¡Las contribuciones son bienvenidas! Por favor, abre un issue o envía un pull request con mejoras o correcciones.

# Licencia
Este proyecto es de código abierto y está disponible bajo la Licencia MIT. (Nota: Especifica una licencia si aplica).

# Créditos
* Creado por csharpkid.
* Powered by GS Innovations


Fecha de última actualización: 06 de marzo de 2025
