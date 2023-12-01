import openai
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# Configura tu clave de API de OpenAI
openai.api_key = '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'
epochs = 10  # Ajusta el número de épocas según tus necesidades

# Cargar datos de ejemplo
textos = ["Hola, ¿cómo estás?", "Me gusta programar en Python", "¿Cuál es tu nombre?"]

# Tokenizar el texto
tokenizer = Tokenizer()
tokenizer.fit_on_texts(textos)
total_palabras = len(tokenizer.word_index) + 1

# Crear secuencias de entrada y salida
input_sequences = []
for texto in textos:
    secuencia = tokenizer.texts_to_sequences([texto])[0]
    for i in range(1, len(secuencia)):
        n_gram_sequence = secuencia[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences para que todas tengan la misma longitud
max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Crear conjuntos de entrada y salida
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# Convertir y a formato one-hot
y = tf.keras.utils.to_categorical(y, num_classes=total_palabras)

# Crear modelo de red neuronal
modelo = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_palabras, 128, input_length=max_sequence_length-1),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(total_palabras, activation='softmax')
])

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelo.summary()

# Función para obtener respuesta de la red neuronal
def obtener_respuesta_red_neuronal(pregunta):
    secuencia = tokenizer.texts_to_sequences([pregunta])[0]
    secuencia = pad_sequences([secuencia], maxlen=max_sequence_length-1, padding='pre')
    predicciones = modelo.predict(secuencia, verbose=0)
    indice_prediccion = tf.argmax(predicciones, axis=1).numpy()[0]
    palabra_predicha = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(indice_prediccion)]
    return palabra_predicha

# Función para obtener respuesta de GPT-3
def obtener_respuesta_gpt3(pregunta):
    intentos = 3
    for intento in range(intentos):
        respuesta = openai.Completion.create(
            engine="text-davinci-003",
            prompt=pregunta,
            max_tokens=50
        )
        if 'choices' in respuesta and respuesta['choices']:
            return respuesta['choices'][0]['text']
        else:
            print(f"Intento {intento + 1} fallido. Reintentando...")
            time.sleep(1)  # Puedes ajustar el tiempo de espera entre reintentos
    return "No se pudo obtener una respuesta válida."

# Bucle para entrenar la red neuronal
for epoch in range(epochs):
    print(f"Época {epoch + 1}/{epochs}")
    modelo.fit(X, y, epochs=1, verbose=0)

# Bucle para hacer preguntas hasta que el usuario decida salir
while True:
    try:
        pregunta = input("Haz una pregunta (máx. 100 caracteres) o escribe 'salir' para terminar: ")
        
        if pregunta.lower() == 'salir':
            print("Saliendo del programa.")
            break
        
        # Obtener respuesta de la red neuronal
        respuesta_red_neuronal = obtener_respuesta_red_neuronal(pregunta)
        print("Respuesta de la red neuronal:", respuesta_red_neuronal)

        # Obtener respuesta de GPT-3
        respuesta_gpt3 = obtener_respuesta_gpt3(pregunta)
        print("Respuesta de GPT-3:", respuesta_gpt3)

        # Verificar consumo de tokens
        if 'usage' in respuesta_gpt3:
            print("Consumo de tokens GPT-3:", respuesta_gpt3['usage']['total_tokens'])

    except Exception as e:
        print(f"Error: {e}")
