import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xplotter.insights import *
import requests
from PIL import Image
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit_shap as st_shap

# Définir l'URL de votre API
#API_URL = "http://localhost:5002/predict"
API_URL ="https://saf-2c3b613dc247.herokuapp.com/predict"

#data=pd.read_csv('C:/Users/Lenovo/application_test.csv')
model = pickle.load(open("model.pkl","rb"))
# Charger les données clients
#clients = pd.read_csv('test_data.csv')
clients= pd.read_csv('test_data.csv', nrows=1000)

############################
# Configuration de la page et définition de styles #
############################
st.set_page_config(
    page_title='Dashboard du Client',
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center; color: black;'>Dashboard du Client</h1>",
    unsafe_allow_html=True,
)

# Centrage de l'image du logo dans la sidebar
col1, col2 = st.columns([1, 1])
with col1:
    st.sidebar.write("")
with col2:
    image = Image.open('16794938722698_Data Scientist-P7-01-banner.png')
    st.sidebar.image(image, use_column_width="always")


# Fonction pour obtenir les prédictions pour un client donné
def get_prediction(client_id):
    params = {'SK_ID_CURR': client_id}
    response = requests.post(API_URL, params=params)
    return response.json()

@st.cache_data
def calcul_valeurs_shap():
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(clients.drop(labels="SK_ID_CURR", axis=1))
    return shap_values

def plot_distribution(selected_features, client_id):
    if selected_features:
        # Afficher le titre
        st.subheader("Distribution des Caractéristiques Sélectionnées")

        # Obtenir les valeurs des caractéristiques pour tous les clients
        feature_values = clients[selected_features]
        client_feature_values = feature_values[clients['SK_ID_CURR'] == client_id].iloc[0]

        fig = go.Figure()

        for selected_feature in selected_features:
            data = clients[selected_feature]

            # Créer un graphique d'histogramme avec Plotly
            histogram = go.Histogram(x=data, name=selected_feature, opacity=0.75)

            # Trouver la valeur de la fonctionnalité pour le client actuel
            client_feature_value = client_feature_values[selected_feature]

            # Calculer les bins pour l'histogramme
            hist_data, bins = np.histogram(data.dropna(), bins=20)

            # Trouver l'indice du bin pour client_feature_value
            client_bin_index = np.digitize(client_feature_value, bins) - 1

            # Créer une liste de couleurs pour les bins
            colors = ["blue"] * (len(bins) - 1)
            if 0 <= client_bin_index < len(colors):  # Vérifier que l'indice est valide
                # Mettre en surbrillance le bin du client sélectionné
                colors[client_bin_index] = "red"

            # Définir la couleur de la barre de l'histogramme
            histogram.marker.color = colors

            # Ajouter le graphique d'histogramme à la figure
            fig.add_trace(histogram)

        # Mettre à jour la mise en page du graphique
        fig.update_layout(
            title_text="Distribution des Caractéristiques Sélectionnées",
            xaxis_title="Valeur de la Caractéristique",
            yaxis_title="Nombre de Clients",
            barmode="overlay",
            title_x=0.5
        )

        # Afficher le graphique
        st.plotly_chart(fig)
if __name__ == "__main__":
    client_id = st.selectbox("Choisir un client", clients['SK_ID_CURR'].tolist())
    if client_id:
        # Obtenir les prédictions pour le client sélectionné
        prediction_data = get_prediction(client_id)

        # Afficher les résultats de prédiction
        st.subheader("Résultats de Prédiction")
        st.write(f"Probabilité de crédit : {prediction_data['probability']:.2f}%")
        decision_pret = prediction_data['Pret']

        # Déterminer la couleur en fonction de la décision de prêt
        if decision_pret == 'accepte':
            couleur_texte = '#008000'  # Vert si le prêt est accepté
        else:
            couleur_texte = '#FF0000'  # Rouge sinon

        # Afficher la décision de prêt avec la couleur spécifique et un format spécial
        st.markdown(
            f'<p style="color: {couleur_texte}; font-size: 18px; font-weight: bold;">Décision de Prêt : {decision_pret}</p>',
            unsafe_allow_html=True
        )

        # Impression du graphique jauge
        boutton = st.button("Afficher le score du client")
        if boutton:
            st.subheader("Score du Client")
            fig = go.Figure(go.Indicator(
                domain={'x': [0, 1], 'y': [0, 1]},
                value=round(prediction_data['probability'], 2),
                mode="gauge+number+delta",
                title={'text': "Le score de ce client", 'font': {'size': 16}},
                delta={'reference': 50, 'increasing': {'color': "#3b203e"}},
                gauge={'axis': {'range': [None, 100],
                            'tickwidth': 3,
                            'tickcolor': 'darkblue'},
                    'bar': {'color': 'white', 'thickness': 0.3},
                    'bgcolor': 'white',
                    'borderwidth': 1,
                    'bordercolor': 'gray',
                    'steps': [{'range': [0, 20], 'color': '#e8af92'},
                            {'range': [20, 40], 'color': '#db6e59'},
                            {'range': [40, 60], 'color': '#b43058'},
                            {'range': [60, 80], 'color': '#772b58'},
                            {'range': [80, 100], 'color': '#3b203e'}],
                    'threshold': {'line': {'color': 'white', 'width': 8},
                                'thickness': 0.8,
                                'value': 35.2}}))

            fig.update_layout(
            paper_bgcolor='white',
            height=400, width=500,
            font={'color': '#772b58', 'family': 'Roboto Condensed'},
            margin=dict(l=30, r=30, b=5, t=5)
                            )
            st.plotly_chart(fig, use_container_width=True)

# Récupération et affichage des informations du client #
    ########################################################
    #data_client=lecture_X_test_original()[lecture_X_test_original().sk_id_curr == client_id]
    #infos_client=data[data['SK_ID_CURR']==client_id]
    infos_client=clients[clients['SK_ID_CURR']==client_id]
    
    col1, col2 = st.columns(2)
    with col1:
        boutton = st.button("Afficher le profil du client")
        if boutton:
        # Titre H2
            st.markdown("""
                    <h2 style="color:#418b85;text-align:center;font-size:1.8em;font-style:italic;font-weight:700;margin:0px;">
                    Profil du client</h2>
                    """, 
                    unsafe_allow_html=True)
            st.write("")
            st.write(f"Genre : **{infos_client['CODE_GENDER'].values[0]}**")
            st.write(f"Aâge : **{(int(infos_client['DAYS_BIRTH'].values[0]/-365))}**")
            st.write(f"Situation familiale : **{infos_client['NAME_FAMILY_STATUS'].values[0]}**")
            st.write(f"Nombre des membres de la famille : **{infos_client['CNT_FAM_MEMBERS'].values[0]}**")
            st.write(f"Revenu total annuel : **{infos_client['AMT_INCOME_TOTAL'].values[0]} $**")
            st.write(f"Type d'emploi : **{infos_client['NAME_INCOME_TYPE'].values[0]}**")
            st.write(f"Type de pret demandé  : **{infos_client['NAME_CONTRACT_TYPE'].values[0]}**")
            st.write(f"Montant du pret demandé  : **{infos_client['AMT_CREDIT'].values[0]} $**")
    
    
     ###############################################################
    # Comparaison du profil du client à son groupe d'appartenance #
    ###############################################################

    # Titre 1
    st.markdown("""
                <br>
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                2. Comparaison du profil du client à celui des clients dont la probabilité de défaut de paiement est proche</h1>
                """, 
                unsafe_allow_html=True)
    # Afficher l'interprétation du score
    st.subheader("Interprétation du Score")
    st.write("Les principales caractéristiques qui influent sur la décision de crédit :")
    feature_names = prediction_data['feature_names']
    shap_values = prediction_data['shap_values']
    interpretation_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': shap_values})
    st.write(interpretation_df)

    # Comparaison avec l'ensemble des clients
    st.subheader("Comparaison avec l'ensemble des clients")
    feature_values=clients.values[0].tolist()
    shap_values = [val[0] if isinstance(val, list) else val for val in shap_values]

    shap_df = pd.DataFrame(
        list(
            zip(
                feature_names,
                shap_values,
                [int(val) for val in feature_values],
            )
        ),
        columns=["Feature", "SHAP Value", "Feature Value"],
    )
    
    positive_features = shap_df.sort_values(by="SHAP Value", ascending=False).head(10)
    negative_features = shap_df.sort_values(by="SHAP Value").head(10)
    #comparison_features = ['feature1', 'feature2', 'feature3']  # Sélectionnez les variables pertinentes
    
    selected_positive_features = st.multiselect("Sélectionnez les caractéristiques influant positivement",
                                                    positive_features)
    selected_negative_features = st.multiselect("Sélectionnez les caractéristiques influant négativement",
                                                    negative_features)
    comparison_data = pd.read_csv("test_data.csv")  # Chargez les données des clients
    
    if st.button('afficher les distributions'):
            plot_distribution(selected_positive_features, client_id)





    shap_values = calcul_valeurs_shap()

# Vérifiez la forme des valeurs SHAP pour vous assurer qu'il s'agit d'une matrice
    st.write(f"Shape of SHAP values: {np.array(shap_values).shape}")

    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Sélectionnez les valeurs SHAP pour la classe positive, si applicable

# Titre 1
    st.markdown("""
            <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
            1. Quelles sont les variables globalement les plus importantes pour comprendre la prédiction ?</h1>
            """, 
            unsafe_allow_html=True)
    st.write("")

    st.write("L’importance des variables est calculée en moyennant la valeur absolue des valeurs de Shap. \
        Les caractéristiques sont classées de l'effet le plus élevé au plus faible sur la prédiction. \
        Le calcul prend en compte la valeur SHAP absolue, donc peu importe si la fonctionnalité affecte \
        la prédiction de manière positive ou négative.")

    st.write("Pour résumer, les valeurs de Shapley calculent l’importance d’une variable en comparant ce qu’un modèle prédit \
        avec et sans cette variable. Cependant, étant donné que l’ordre dans lequel un modèle voit les variables peut affecter \
        ses prédictions, cela se fait dans tous les ordres possibles, afin que les fonctionnalités soient comparées équitablement. \
        Cette approche est inspirée de la théorie des jeux.")

    st.write("*__Le diagramme d'importance des variables__* répertorie les variables les plus significatives par ordre décroissant.\
        Les *__variables en haut__* contribuent davantage au modèle que celles en bas et ont donc un *__pouvoir prédictif élevé__*.")
    if st.button("Afficher le diagramme d'importance des variables"):

        fig = plt.figure()
        plt.title("Interprétation Globale :\n Diagramme d'Importance des Variables", 
        fontname='Roboto Condensed',
        fontsize=20, 
        fontstyle='italic')
        st_shap(shap.summary_plot(shap_values, 
                        feature_names=clients.drop(labels="SK_ID_CURR", axis=1).columns,
                        plot_size=(12, 16),
                        color='#9ebeb8',
                        plot_type="bar",
                        max_display=56,
                        show=False))
        plt.show()

# Titre 2
    st.markdown("""
            <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
            2. Quel est l'Impact de chaque caractéristique sur la prédiction ?</h1>
            """, 
            unsafe_allow_html=True)
    st.write("")
    st.write("Le diagramme des valeurs SHAP ci-dessous indique également comment chaque caractéristique impacte la prédiction. \
        Les valeurs de Shap sont représentées pour chaque variable dans leur ordre d’importance. \
        Chaque point représente une valeur de Shap (pour un client).")
    st.write("Les points fuchsia représentent des valeurs élevées de la variable et les points verts des valeurs basses de la variable.")
    if st.button("Afficher le diagramme des valeurs SHAP"):

        fig = plt.figure()
        plt.title("Interprétation Globale :\n Impact de chaque caractéristique sur la prédiction\n", 
        fontname='Roboto Condensed',
        fontsize=20, 
        fontstyle='italic')
        st_shap(shap.summary_plot(shap_values, 
                     features=clients.drop(labels="SK_ID_CURR", axis=1),
                     feature_names=clients.drop(labels="SK_ID_CURR", axis=1).columns,
                     plot_size=(12, 16),
                     color='#9ebeb8',
                     plot_type="bar",
                     max_display=56,
                     show=False))
        plt.show()

    st.write("14 variables ont un impact significatif sur la prédiction (Moyenne des valeurs absolues des valeurs de Shap >= 0.1). \
        La première est sans contexte le score normalisé à partir d'une source de données externes.")
    st.markdown("""
    1. Plus la valeur du 'Score normalisé à partir d'une source de données externe' est faible (points de couleur vert), 
   et plus la valeur Shap est élevée et donc plus le modèle prédit que le client aura des difficultés de paiement.<br>
   2. Plus la dernière demande de crédit du client, avant la demande actuelle, enregistrée au bureau des crédits, est récente 
   (points de couleur vert), plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
   3. Plus le montant payé par le client par rapport au montant attendu est faible (points de couleur vert), 
   plus la valeur Shap est élevée et donc plus le modèle pédit que le client aura des difficultés de paiement.<br>
   4. Si le client est un homme, la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
   5. Plus la durée mensuelle du contrat précédent du client est élevée (points de couleur fuchsia), 
   plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
   6. Plus le nombre de contrats précédents refusés pour le client est élevé (points de couleur fuchsia), 
   plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
   7. Plus le client est jeune (points de couleur vert), plus la valeur Shap est élevée et
   donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
   8. Lorsque le client n'est pas allé dans l'enseignement supérieur (points vert), 
   la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
9. Nombre de crédits soldés du client enregistrés au bureau du crédit : *impact indéfini* <br>
    10. Plus le nombre de versements réalisés par la client est faible (points de couleur vert), 
    plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.
    11. Plus l'ancienneté du client dans son entreprise est faible (points de couleur vert), 
    plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.
12. Plus le nombre de Cartes de Crédit du client enregistrées au bureau du crédit est élevé (points de couleur fuchsia),
    plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.
13. Plus le montant de la demande de prêt actuelle du client est élevé (points de couleur fuchsia), 
    plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.
14. Plus le montant de la demande de prêt précédente du client est faible (points de couleur vert), 
    plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.
        """, 
        unsafe_allow_html=True)

# Titre 2
    st.markdown("""
            <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
            2. Graphique de dépendance</h1>
            """, 
            unsafe_allow_html=True)
    st.write("Nous pouvons obtenir un aperçu plus approfondi de l'effet de chaque fonctionnalité \
          sur l'ensemble de données avec un graphique de dépendance.")
    st.write("Le dependence plot permet d’analyser les variables deux par deux en suggérant une possiblité d’observation des interactions.\
          Le scatter plot représente une dépendence entre une variable (en x) et les shapley values (en y) \
          colorée par la variable la plus corrélées.")

################################################################################
# Création et affichage du sélecteur des variables et des graphs de dépendance #
################################################################################
    liste_variables = clients.drop(labels="SK_ID_CURR", axis=1).columns.to_list()

    col1, col2 = st.columns(2)  # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
    with col1:
        var_id = st.selectbox("*Veuillez sélectionner une variable à l'aide du menu déroulant 👇*", 
                          liste_variables)
        st.write("Vous avez sélectionné la variable :", var_id)

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(121)
    shap.dependence_plot(var_id, 
                    shap_values, 
                    clients.drop(labels="SK_ID_CURR", axis=1), 
                    interaction_index=None,
                    alpha=0.5,
                    x_jitter=0.5,
                    title="Graphique de Dépendance",
                    ax=ax1,
                    show=False)
    ax2 = fig.add_subplot(122)
    shap.dependence_plot(var_id, 
                    shap_values, 
                    clients.drop(labels="SK_ID_CURR", axis=1), 
                    interaction_index='auto',
                    alpha=0.5,
                    x_jitter=0.5,
                    title="Graphique de Dépendance et Intéraction",
                    ax=ax2,
                    show=False)
    fig.tight_layout()
    st.pyplot(fig)
