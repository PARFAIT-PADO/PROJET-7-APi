from flask import Flask, request, jsonify
import pandas as pd
import joblib 
import shap

# Initialisation de l'application Flask
app = Flask(__name__)

# Chemin vers les données et le modèle


# Charger le modèle
#joblib.dump(model, 'new_model.pkl')
mon_best_model = joblib.load("PROJECT_7_MODEL")


# Charger les datasets
X_test = pd.read_csv('X_test_feat_new.csv')

# Initialisation de SHAP
explainer = shap.TreeExplainer(mon_best_model)
shap_values = explainer.shap_values(X_test)

# Route d'accueil pour vérifier l'état de l'API
@app.route('/')
def home():
    return "API pour prédictions de crédits est en ligne!"

# Route pour obtenir la prédiction d'un client en fonction de son identifiant
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    client_id = data.get('client_id')

    # Vérifier si l'identifiant est valide
    if client_id is None or client_id < 0 or client_id >= X_test.shape[0]:
        return jsonify({'error': 'Identifiant client invalide!'}), 400
    
    # Prédictions pour le client donné
    y_pred_model_proba = mon_best_model.predict_proba(X_test)
    prob_default = round(y_pred_model_proba[client_id][1] * 100, 2)

    # Détermination de la décision
    if prob_default >= 20.0:
        decision = "REFUSÉE"
    else:
        decision = "ACCEPTÉE"
    
    # Retourner la prédiction et la décision
    result = {
        'client_id': client_id,
        'prob_default': prob_default,
        'decision': decision
    }
    
    return jsonify(result)



if __name__ == '__main__':
    app.run(debug=True)
