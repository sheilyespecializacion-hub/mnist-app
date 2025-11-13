from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(
    title="MNIST Classifier API",
    description="API REST para clasificar dígitos escritos a mano usando un modelo entrenado con TensorFlow (MNIST).",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # puedes restringirlo luego
    allow_methods=["*"],
    allow_headers=["*"],
)

# contador simple de predicciones (métrica básica de uso)
app.state.prediction_count = 0

# === Cargar modelo ===
try:
    model = tf.keras.models.load_model("mnist_cnn.keras")
    print("✅ Modelo cargado correctamente.")
except Exception as e:
    print("❌ Error cargando el modelo:", e)
    model = None


def preprocess(img_bytes: bytes) -> np.ndarray:
    """
    Convierte la imagen a escala de grises, la redimensiona a 28x28,
    normaliza a [0,1] y deja el shape (1,28,28,1), como en el entrenamiento.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # escala de grises
    img = img.resize((28, 28))

    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=-1)   # (28,28) -> (28,28,1)
    arr = np.expand_dims(arr, axis=0)    # (28,28,1) -> (1,28,28,1)
    return arr


# ========= FRONTEND: Página web en "/" =========
@app.get("/", response_class=HTMLResponse)
async def root():
    # NOTA: aquí va todo el HTML + JS del frontend
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
      <meta charset="UTF-8" />
      <title>Clasificador MNIST</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          text-align: center;
          margin-top: 30px;
        }
        #canvas {
          border: 2px solid #333;
          background: black;
          cursor: crosshair;
        }
        button {
          margin: 10px;
          padding: 8px 16px;
          font-size: 14px;
        }
        #history {
          max-width: 400px;
          margin: 20px auto;
          text-align: left;
          border: 1px solid #ddd;
          padding: 10px;
          border-radius: 6px;
        }
        #history h4 {
          margin-top: 0;
        }
        #history ul {
          padding-left: 20px;
        }
      </style>
    </head>
    <body>
      <h2>Clasificador de dígitos MNIST 🧠✏️</h2>
      <p>Dibuja un número (0–9) en blanco sobre fondo negro.</p>

      <canvas id="canvas" width="200" height="200"></canvas><br>

      <button onclick="clearCanvas()">Limpiar</button>
      <button onclick="sendImage()">Clasificar</button>

      <h3 id="result"></h3>

      <div id="metrics">
        <p><strong>Predicciones en esta sesión:</strong> <span id="count">0</span></p>
      </div>

      <div id="history">
        <h4>Historial de predicciones</h4>
        <ul id="history-list"></ul>
      </div>

      <script>
        const API_URL = "/predict";  // mismo servicio, mismo host

        const canvas = document.getElementById("canvas");
        const ctx = canvas.getContext("2d");

        ctx.lineWidth = 18;
        ctx.lineCap = "round";
        ctx.strokeStyle = "white";

        let drawing = false;
        let predictionCount = 0;

        // --- PC (mouse) ---
        canvas.addEventListener("mousedown", () => drawing = true);
        canvas.addEventListener("mouseup", () => drawing = false);
        canvas.addEventListener("mouseleave", () => drawing = false);
        canvas.addEventListener("mousemove", drawMouse);

        function drawMouse(e) {
          if (!drawing) return;
          const rect = canvas.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const y = e.clientY - rect.top;
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(x + 0.1, y + 0.1);
          ctx.stroke();
        }

        // --- Táctil (móvil) ---
        canvas.addEventListener("touchstart", (e) => {
          e.preventDefault();
          drawing = true;
        });
        canvas.addEventListener("touchend", (e) => {
          e.preventDefault();
          drawing = false;
        });
        canvas.addEventListener("touchmove", (e) => {
          e.preventDefault();
          if (!drawing) return;
          const rect = canvas.getBoundingClientRect();
          const touch = e.touches[0];
          const x = touch.clientX - rect.left;
          const y = touch.clientY - rect.top;
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(x + 0.1, y + 0.1);
          ctx.stroke();
        });

        function clearCanvas() {
          ctx.fillStyle = "black";
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          document.getElementById("result").innerText = "";
        }
        clearCanvas();

        function addToHistory(digit, confidence) {
          const ul = document.getElementById("history-list");
          const li = document.createElement("li");
          li.textContent = `Predicción: ${digit} (confianza: ${confidence.toFixed(3)})`;
          ul.prepend(li); // agrega al inicio
        }

        function updateCount() {
          predictionCount += 1;
          document.getElementById("count").innerText = predictionCount;
        }

        async function sendImage() {
          document.getElementById("result").innerText = "Clasificando...";

          const blob = await new Promise(resolve =>
            canvas.toBlob(resolve, "image/png")
          );

          const formData = new FormData();
          formData.append("file", blob, "digit.png");

          try {
            const res = await fetch(API_URL, {
              method: "POST",
              body: formData,
            });

            if (!res.ok) {
              const text = await res.text();
              document.getElementById("result").innerText =
                "Error en la predicción: " + res.status + " " + text;
              return;
            }

            const data = await res.json();
            document.getElementById("result").innerText =
              `Predicción: ${data.digit} (confianza: ${data.confidence.toFixed(3)})`;

            addToHistory(data.digit, data.confidence);
            updateCount();
          } catch (err) {
            console.error(err);
            document.getElementById("result").innerText =
              "Error al conectar con la API";
          }
        }
      </script>
    </body>
    </html>
    """


# ========= ENDPOINT DE SALUD =========
@app.get("/health")
async def health():
    if model is None:
        return {"status": "error", "detail": "modelo no cargado"}
    return {"status": "ok", "detail": "servicio funcionando"}


# ========= ENDPOINT DE PREDICCIÓN =========
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validaciones básicas
    if file is None:
        raise HTTPException(status_code=400, detail="No se envió ningún archivo.")

    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen PNG o JPG.")

    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado en el servidor.")

    try:
        img_bytes = await file.read()
        x = preprocess(img_bytes)
        preds = model.predict(x)
        digit = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        # incrementar métrica simple de uso
        app.state.prediction_count += 1
        print(f"Predicción #{app.state.prediction_count}: dígito={digit}, confianza={confidence:.3f}")

        return {
            "digit": digit,
            "confidence": confidence,
            "probabilities": preds[0].tolist(),
            "prediction_number": app.state.prediction_count
        }
    except HTTPException:
        raise
    except Exception as e:
        print("Error en /predict:", e)
        raise HTTPException(status_code=500, detail="Error interno al procesar la imagen.")
