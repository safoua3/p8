#https://safoua-ea72e808f916.herokuapp.com/predict?id=
import os
import pickle
import pandas as pd
import shap
from flask import Flask, jsonify, request
import json

app = Flask(__name__)

# Charger le modèle en dehors de la clause if __name__ == "__main__":
#model_path = os.path.join(current_directory, "..", "Simulations", "Best_model", "model.pkl")
model = pickle.load(open("model.pkl","rb"))
@app.route("/predict", methods=['GET','POST'])
def predict():
    
    id =int(request.args.get('SK_ID_CURR'))
    if id==None :
        return "cet identifiant n'existe pas"
     
    else:
        print(str(id))
        df = pd.read_csv("test_data.csv")
        sample = df.loc[df['SK_ID_CURR']==id]
        #print(sample)
        sample = sample.drop(columns=['SK_ID_CURR'])
        #print(sample)
        proba = model.predict_proba(sample)[:, 1][0]
        #proba = prediction[0][1]
        print(proba)
        seuil=0.63
        if proba >= seuil:
            Pret = "Accepté"
        else:
            Pret = "Refusé"
        
         # Calculer les valeurs SHAP pour l'échantillon donné
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
    
    # Retourner les valeurs SHAP avec la probabilité
        return jsonify({
        'probability': proba*100, 
        'Pret': Pret,
        'shap_values': shap_values[1][0].tolist(),
        #'shap_values': shap_values[0].tolist(),  # Utilisez shap_values[0] pour la classe principale (première classe)
        'feature_names': sample.columns.tolist(),
        'feature_values': sample.values[0].tolist()
    })
        
    
    


if __name__ == '__main__':
    app.run(debug=True,port=5002)

