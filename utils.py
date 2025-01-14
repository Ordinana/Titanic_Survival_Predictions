import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini = genai.GenerativeModel("gemini-2.0-flash-exp")

# Define la función para generar el prompt 
def get_prompt(new_prediction):
    prompt = f"""Estoy haciendo una API de prediccion de supervivencia con el dataset del titanic.
    He usado 3 features:
    Pclass (Clase del billete) = {new_prediction['pclass']},
    Sex = {new_prediction['sex']} (0 = Femenino, 1 = Masculino), 
    Age = {int(new_prediction['age'])} (años),
    Quiero que tengas en cuenta estas 3 variables y principalmente la prediccion ({new_prediction['prediction']}), la cual dirá si el parajero sobrevive (1), o no (0).
    A si que, en función de si sobrevive o no, genera un texto breve, narrativo, explicando porqué ha sobrevivido o porqué no.   

    IMPORTANTE: No incluyas metadatos, formato enriquecido ni texto innecesario.
    IMPORTANTE: El formato de salida ha de ser unica y exclusivamente el texto narrado, no me des saludos ni metadatos innecesarios.
    IMPORTANTE: Sé breve,entre 30 y 70 palabras.

    Quiero que generes un texto narrativo coherente que explique esta predicción. Ten en cuenta lo siguiente:
    1. Si la predicción es 1 (Sobrevivió), genera un texto optimista, gracioso o motivador, que describa cómo esta persona logró sobrevivir.
    2. Si la predicción es 0 (No sobrevivió), genera un texto más melancólico pero con humor negro, que explique por qué esta persona no logró sobrevivir.
    3. Usa los datos de entrada (Pclass, Sex, Age) para construir la narrativa. Por ejemplo, menciona la clase del billete, la edad o el sexo de la persona en la historia.
    4. El texto debe ser directamente relevante para la predicción. No contradigas el resultado (0: No sobrevivió, 1: Sobrevivió).
    5. Sé siempre creativo, con sentido del humor, gracioso, incluso si ves la oportunidad di algun chiste malo o cuenta una mini historia o moraleja basada en los datos.

    Ejemplo de salida:
    Si Prediction = 1:
    "La joven de primera clase, con 22 años y un vestido elegante, logró subirse a uno de los botes salvavidas justo a tiempo. Dicen que su instinto de supervivencia fue digno de aplausos. ¡Qué suerte que viajaba en primera clase!"

    Si Prediction = 0:
    "El caballero de tercera clase, con 30 años y grandes sueños, no pudo escapar del destino del Titanic. Al menos dejó su sombrero como recuerdo para la posteridad. ¡Qué injusto puede ser el Atlántico a veces!"

    Por favor, genera un único ejemplo de texto para la predicción dada."""
    return prompt


# Define la función para generar texto con parámetros ajustables
def get_text(gemini, prompt, temperatura=0.7, max_output_tokens=150, top_p=1.0, top_k=40):
  """
  Genera texto usando el modelo Gemini con parámetros ajustables.

  Args:
    prompt: El prompt (instrucción) para el modelo.
    temperatura: Controla la aleatoriedad del texto generado (valores entre 0.0 y 1.0).
                 Un valor más bajo hace que el texto sea más determinista (predecible).
                 Un valor más alto hace que el texto sea más creativo (aleatorio).
    max_output_tokens: El número máximo de tokens a generar en la respuesta.
    top_p: Muestreo de núcleo. El modelo elige de los tokens que conforman el 'top p' de probabilidad.
           Valores entre 0.0 y 1.0.
    top_k: Muestreo de top-k. El modelo elige de los 'top-k' tokens más probables.

  Returns:
    El texto generado por el modelo.
  """
  generation_config = {
        "temperature": temperatura,
        "max_output_tokens": max_output_tokens,
        "top_p": top_p,
        "top_k": top_k
  }

  response = gemini.generate_content(prompt, generation_config=generation_config)
  return response.text
