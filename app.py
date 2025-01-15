from flask import Flask, request, jsonify, render_template
import pickle
import datetime
from sqlalchemy import create_engine, inspect
import pandas as pd
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Configura el backend no interactivo
import matplotlib.pyplot as plt
import json
import base64
import os
from dotenv import load_dotenv
import google.generativeai as genai
from utils import get_prompt, get_text

load_dotenv()

print(os.environ)

app = Flask(__name__)

# PARA SQLAlquemy:

### postgresql://user:password@host:5432/postgres
### mysql://user:password@host:3306/mydb
###  --------->>>  sqlite:///titanic.db


# Cargar el modelo entrenado
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

### HABRÍA QUE CARGAR LA NORMALIZACIÓN

# Crear motor de SQLAlchemy
cadena = os.environ["CADENA"]
engine = create_engine(cadena)

@app.route('/', methods=['GET'])
def home():
    return render_template("formulario.html")


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint que recibe datos del HTML, realiza una predicción y guarda los datos en la base de datos.
    """

    # 1. Extraer los datos de entrada del HTLM
    pclass = int(request.form.get('Pclass'))
    sex = int(request.form.get('Sex'))
    age = int(request.form.get('Age'))
    # Se podria hacer asi (en conjunto): inputs = [int(x) for x in request.form.values()]

    # 2. Realizar predicción con el modelo

    ### HABRÍA QUE APLICAR LA NORMALIZACIÓN

    # Completa aquí: usa model.predict()
    prediction = model.predict([[pclass, sex, age]])
    # Alternativa para extraer el resultado del array: model.predict([[pclass, sex, age]])[0]

    # 3. Guardar en la base de datos
    timestamp = datetime.datetime.now().isoformat()
    # Completa aquí: inserta los datos (inputs, predicción, timestamp) en la base de datos
    # Monta DF para subir
    new_prediction = pd.DataFrame({"pclass": [pclass],
                                    "sex": [sex],
                                    "age":[age],
                                    "prediction": [int(prediction[0])],
                                    "timestamp": [timestamp[0:19]]})

    # Si usas la alternativa anterior: prediction: int(prediction)
    # Subir la prediccion
    new_prediction.to_sql("predictions", con=engine, if_exists='append', index=False)

    ### Generamos la gráfica
    read_predictions = pd.read_sql('''SELECT * FROM predictions''', con=engine)
    fig = plt.figure()

    # Obtener conteo de predicciones
    value_counts = read_predictions.prediction.value_counts()

    # Asignar colores en función del índice
    colors = ['red' if index == 0 else 'green' for index in value_counts.index]

    # Crear gráfico de barras
    ax = value_counts.plot(kind="bar", color=colors)

    # Cambiar las etiquetas del eje X
    ax.set_xticks([1, 0])  # Posiciones de las barras
    ax.set_xticklabels(["Sobrevive", "No sobrevive"])

    # Personalizar título y etiquetas
    plt.title("Histórico de predicciones")
    plt.xlabel('')  # Limpia la etiqueta del eje X
    plt.xlabel('')  # Limpia la etiqueta del eje Y

    # Ajustar diseño para que no se corten elementos
    plt.tight_layout()
    
    # Agregar etiquetas de los valores encima de las barras
    # for index, value in enumerate(value_counts):
    #     ax.text(index, value + 0.5, str(value), ha='center', va='top', fontsize=10)

    # Guardar la gráfica en un buffer en memoria
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)

    # Codificar la imagen para pasarla por JSON a los resultados
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    ### -----> GENERAR TEXTO IA
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gemini = genai.GenerativeModel("gemini-2.0-flash-exp")
    prompt = get_prompt(new_prediction)
    generacion = get_text(gemini, prompt)

    # Devolver el resultado, la imagen (grafica) y el texto generado por gemini como respuesta
    return render_template("resultado.html", prediccion=prediction, grafica=img_base64, gen_text=generacion)
    

@app.route('/records', methods=['GET'])
def records():
    """
    Endpoint que devuelve todos los registros guardados en la base de datos.
    """
    read_predictions = pd.read_sql("'''SELECT * FROM predictions'''", con=engine)
    return json.loads(read_predictions.to_json(orient="records"))


if __name__ == "__main__": # Con esto Python sabrá que este fichero es un Script.py ejecutable y no un modulo (Se utiliza si se va a desplegar como APP)
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)