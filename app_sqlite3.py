from flask import Flask, request, jsonify, Response
import pickle
import datetime
import sqlite3
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt

app = Flask(__name__)

# Si se utilizase SQLAlquemy (Que seria más correcto):

### postgresql://user:password@host:5432/postgres
### mysql://user:password@host:3306/mydb
###  --------->>>  sqlite:///titanic.db

# Pero en éste caso se utiliza directamente sqlite3...

# Cargar el modelo entrenado
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

### HABRÍA QUE CARGAR LA NORMALIZACIÓN

# Inicializar la base de datos
def init_db():
    connection = sqlite3.connect("./titanic.db") # SI ESTÁ CONECTA Y SI NO, LA CREA
    crsr = connection.cursor()
    # Conectar a la base de datos y crear la tabla 'predictions' si no existe
    # Completa aquí: conexión SQLite y creación de tabla con campos (inputs, prediction, timestamp)
    query = '''
    CREATE TABLE IF NOT EXISTS predictions (
        pclass INTEGER,         -- Clase del pasajero (entero)
        sex INTEGER,            -- Sexo (codificado como entero)
        age REAL,               -- Edad (número decimal)
        prediction INTEGER,     -- Predicción (entero)
        timestamp TIMESTAMP     -- Marca temporal (fecha y hora)
    );
    '''
    crsr.execute(query)
    crsr.close()
    connection.close()

init_db()


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint que recibe datos en formato JSON, realiza una predicción y guarda los datos en la base de datos.
    """
    connection = sqlite3.connect("./titanic.db")
    crsr = connection.cursor()

    try:
        data = request.json
        # 1. Extraer los datos de entrada del JSON recibido
        pclass = int(data.get('Pclass'))
        sex = int(data.get('Sex'))
        age = int(data.get('Age'))

        # 2. Realizar predicción con el modelo

        ### HABRÍA QUE APLICAR LA NORMALIZACIÓN

        # Completa aquí: usa model.predict()
        prediction = model.predict([[pclass, sex, age]])
        # Alternativa para extraer el resultado del array: model.predict([[pclass, sex, age]])[0]

        # 3. Guardar en la base de datos
        timestamp = datetime.datetime.now().isoformat()
        # Completa aquí: inserta los datos (inputs, predicción, timestamp) en la base de datos
        query = '''INSERT INTO predictions (pclass, sex, age, prediction, timestamp)
           VALUES (?, ?, ?, ?, ?);
        '''
        # Lanza la consulta
        crsr.execute(query, (pclass, sex, age, int(prediction[0]), timestamp[0:19]))
        # Si usas la alternativa anterior: crsr.execute(query, (pclass, sex, age, int(prediction), timestamp))

        # Confirmar los cambios
        connection.commit()
        connection.close()

        return jsonify({"prediction": int(prediction), "timestamp": timestamp})
    
    except Exception as e:
        return jsonify({"error": str(e), "pclas":pclass, "sex": sex , "age": age})
    

@app.route('/records', methods=['GET'])
def records():
    """
    Endpoint que devuelve todos los registros guardados en la base de datos.
    """
    try:
        # Conectar a la base de datos y recuperar los registros
        connection = sqlite3.connect("./titanic.db")
        connection.row_factory = sqlite3.Row  # Permitir acceso a los resultados como diccionarios
        crsr = connection.cursor()
        query = '''SELECT * FROM predictions;'''
        crsr.execute(query)

        records = crsr.fetchall()  # Sustituir por los datos recuperados de la base de datos

        # Convertir a una lista de diccionarios
        result = [dict(row) for row in records]
        connection.close()

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/plot', methods=['GET'])
def plot():
    """
    Endpoint que genera una gráfica a partir de los datos en la base de datos y la devuelve como imagen.
    """
    import numpy as np
    try:
        connection = sqlite3.connect("./titanic.db")
        connection.row_factory = sqlite3.Row
        crsr = connection.cursor()

        query = '''SELECT age, prediction FROM predictions;'''
        crsr.execute(query)
        records = crsr.fetchall()
        connection.close()

        # Crear un DataFrame asegurando los nombres correctos
        import pandas as pd
        df = pd.DataFrame(records, columns=['age', 'prediction'])

        # Ordenar los datos por edad para que las líneas tengan sentido
        df = df.sort_values(by='age')

        # Crear el gráfico de líneas
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(df['age'], df['prediction'], marker='o', linestyle='-', color='blue', label='Predictions')
        plt.title('Predicciones por Edad')
        plt.xlabel('Edad')
        plt.ylabel('Predicción')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Guardar la gráfica en un buffer en memoria
        from io import BytesIO
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()

        # Devolver la imagen como respuesta
        return Response(buffer, mimetype='image/png')
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__": # Con esto Python sabrá que este fichero es un Script.py ejecutable y no un modulo (Se utiliza si se va a desplegar como APP)
    app.run(debug=True)