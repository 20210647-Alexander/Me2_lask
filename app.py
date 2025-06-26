from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Cargar modelo y scaler
model = joblib.load('modelo_mlp.pkl')
scaler = joblib.load('scaler.pkl')

# Ruta principal
@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None

    if request.method == 'POST':
        try:
            # Obtener valores del formulario
            data = [
                float(request.form['Minutes']),
                float(request.form['Bonus']),
                float(request.form['ICT']),
                float(request.form['TSB']),
                float(request.form['Influence']),
                float(request.form['Goals_scored']),
            ]

            # Escalar datos
            scaled_data = scaler.transform([data])

            # Predecir
            prediction = model.predict(scaled_data)[0]
            prediction = round(prediction, 2)

        except Exception as e:
            prediction = f"Error en la predicción: {e}"

    return render_template('formulario.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
