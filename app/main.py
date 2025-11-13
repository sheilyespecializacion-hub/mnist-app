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
    return """
    <!DOCTYPE html>
    <html lang='es'>
    <head>
      <meta charset='UTF-8' />
      <title>Aprende los números</title>
      <style>
        * {
          box-sizing: border-box;
        }

        body {
          margin: 0;
          font-family: "Comic Sans MS", "Poppins", system-ui, sans-serif;
          background: linear-gradient(135deg, #fef3c7, #bfdbfe);
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 20px;
        }

        .app {
          width: 100%;
          max-width: 960px;
          background: #ffffff;
          border-radius: 24px;
          box-shadow: 0 18px 40px rgba(0, 0, 0, 0.15);
          padding: 22px 20px 24px;
          border: 4px solid #fbbf24;
        }

        .header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 16px;
        }

        .title-left {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .title-left h1 {
          margin: 0;
          font-size: 1.8rem;
          color: #1f2937;
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .title-left h1 span {
          font-size: 2rem;
        }

        .subtitle {
          font-size: 0.95rem;
          color: #4b5563;
        }

        .chip {
          background: #fef9c3;
          border-radius: 999px;
          padding: 6px 12px;
          font-size: 0.8rem;
          color: #92400e;
          border: 1px dashed #f59e0b;
        }

        .layout {
          display: grid;
          grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
          gap: 18px;
        }

        .card {
          background: #eff6ff;
          border-radius: 20px;
          padding: 14px 14px 16px;
          border: 2px solid #bfdbfe;
        }

        .card:nth-child(1) {
          background: #e0f2fe;
        }

        .card h2 {
          margin: 0 0 4px;
          font-size: 1.1rem;
          color: #1f2937;
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .card p {
          margin: 0 0 10px;
          font-size: 0.9rem;
          color: #4b5563;
        }

        #canvas {
          border-radius: 18px;
          border: 3px solid #3b82f6;
          background: #020617;
          cursor: crosshair;
          box-shadow: 0 10px 25px rgba(37, 99, 235, 0.35);
        }

        .controls {
          display: flex;
          justify-content: center;
          gap: 12px;
          margin-top: 10px;
          flex-wrap: wrap;
        }

        button {
          border: none;
          border-radius: 999px;
          padding: 9px 18px;
          font-size: 0.95rem;
          font-weight: 600;
          cursor: pointer;
          display: inline-flex;
          align-items: center;
          gap: 6px;
          transition: transform 0.08s ease, box-shadow 0.1s ease, background 0.1s ease;
        }

        .btn-primary {
          background: #22c55e;
          color: #052e16;
          box-shadow: 0 10px 18px rgba(34, 197, 94, 0.4);
        }

        .btn-primary:hover {
          background: #16a34a;
          transform: translateY(-1px);
          box-shadow: 0 14px 24px rgba(22, 163, 74, 0.5);
        }

        .btn-secondary {
          background: #f97316;
          color: #fff7ed;
          box-shadow: 0 10px 18px rgba(249, 115, 22, 0.4);
        }

        .btn-secondary:hover {
          background: #ea580c;
          transform: translateY(-1px);
          box-shadow: 0 14px 24px rgba(234, 88, 12, 0.5);
        }

        .result-main {
          margin-top: 4px;
          padding: 8px 10px;
          background: #fef3c7;
          border-radius: 12px;
          border: 2px dashed #f59e0b;
        }

        .result-label {
          font-size: 0.9rem;
          color: #92400e;
        }

        #result {
          margin-top: 4px;
          font-size: 1.4rem;
          font-weight: 700;
          color: #b91c1c;
        }

        .metric {
          margin-top: 10px;
          font-size: 0.9rem;
          color: #111827;
        }

        .metric span {
          font-weight: 700;
          color: #2563eb;
        }

        #history {
          margin-top: 10px;
          padding-top: 6px;
          border-top: 2px dotted #93c5fd;
        }

        #history h4 {
          margin: 0 0 6px;
          font-size: 0.95rem;
          color: #1d4ed8;
          display: flex;
          align-items: center;
          gap: 6px;
        }

        #history-list {
          margin: 0;
          padding-left: 18px;
          font-size: 0.9rem;
          max-height: 150px;
          overflow-y: auto;
          color: #1f2937;
        }

        #history-list li {
          margin-bottom: 3px;
        }

        .footer-note {
          margin-top: 10px;
          font-size: 0.8rem;
          color: #6b7280;
        }

        @media (max-width: 780px) {
          .app {
            padding: 16px 14px 18px;
          }
          .layout {
            grid-template-columns: minmax(0, 1fr);
          }
        }
      </style>
    </head>
    <body>
      <div class='app'>
        <div class='header'>
          <div class='title-left'>
            <h1><span>🎨</span> ¡Juega con los números!</h1>
            <p class='subtitle'>
              Niños y niñas pueden practicar cómo escribir los números del <strong>0</strong> al <strong>9</strong>.
            </p>
          </div>
          <div class='chip'>
            🤖 Modelo de inteligencia artificial con MNIST
          </div>
        </div>

        <div class='layout'>
          <!-- Lado izquierdo: Canvas -->
          <div class='card'>
            <h2>✏️ Dibuja tu número</h2>
            <p>
              Usa el lápiz para escribir un número grande y clarito.
              ¡Luego aprieta el botón verde para que el robot adivine!
            </p>
            <canvas id='canvas' width='260' height='260'></canvas>
            <div class='controls'>
              <button class='btn-secondary' onclick='clearCanvas()'>
                🧽 Borrar
              </button>
              <button class='btn-primary' onclick='sendImage()'>
                🤖 Adivinar número
              </button>
            </div>
          </div>

          <!-- Lado derecho: Resultados -->
          <div class='card'>
            <h2>🌟 Resultado del robot</h2>
            <p>
              Aquí ves qué número cree el robot que escribiste y un listado con tus intentos.
            </p>

            <div class='result-main'>
              <div class='result-label'>El robot dice que escribiste:</div>
              <div id='result'>Dibuja un número y pulsa "Adivinar número".</div>
            </div>

            <div class='metric'>
              Veces que has practicado: <span id='count'>0</span>
            </div>

            <div id='history'>
              <h4>📚 Tus intentos</h4>
              <ul id='history-list'></ul>
            </div>

            <div class='footer-note'>
              Recomendado para niños de 3 a 5 años. Siempre con la compañía de un adulto. 💛
            </div>
          </div>
        </div>
      </div>

      <script>
        const API_URL = "/predict";  // mismo servicio, mismo host

        const canvas = document.getElementById("canvas");
        const ctx = canvas.getContext("2d");

        ctx.lineWidth = 22;
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

        // --- Táctil (móvil / tablet) ---
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
          document.getElementById("result").innerText =
            'Dibuja un número y pulsa "Adivinar número".';
        }

        clearCanvas();

        function addToHistory(digit, confidence) {
          const ul = document.getElementById("history-list");
          const li = document.createElement("li");
          li.textContent = `Número: ${digit}  (confianza: ${confidence.toFixed(2)})`;
          ul.prepend(li);
        }

        function updateCount() {
          predictionCount += 1;
          document.getElementById("count").innerText = predictionCount;
        }

        async function sendImage() {
          const resultEl = document.getElementById("result");
          resultEl.innerText = "El robot está pensando... 🤖💭";

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
              resultEl.innerText =
                "Ups, hubo un error: " + res.status + " " + text;
              return;
            }

            const data = await res.json();
            resultEl.innerText =
              `¡Creo que es el número ${data.digit}! 🎉 (confianza: ${data.confidence.toFixed(2)})`;

            addToHistory(data.digit, data.confidence);
            updateCount();
          } catch (err) {
            console.error(err);
            resultEl.innerText =
              "No pudimos hablar con el robot. Intenta otra vez. 🛠️";
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

