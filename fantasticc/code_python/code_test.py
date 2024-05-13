import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Chargement des données du catalogue
catalog_data = pd.read_excel("Catalogue.xlsx")

# Chargement des données des dimensions des magasins
dimension_data = pd.read_excel("Dimention_magazin.xlsx")

# Chargement des données de ventes
sales_data = pd.read_excel("fais-DATA.xlsx")

# Fusion des données
merged_data = pd.merge(sales_data, catalog_data, left_on='FK_Livre', right_on='PK_Livre')
merged_data = pd.merge(merged_data, dimension_data, left_on='FK_Magazin', right_on='PK_Magazin')

# Conversion de la colonne Date_Ticket en datetime
merged_data['Date_Ticket'] = pd.to_datetime(merged_data['Date_Ticket'])

# Extraction de l'année à partir de la colonne Date_Ticket
merged_data['Year'] = merged_data['Date_Ticket'].dt.year

# Exploration des données
print(merged_data.head())

# Visualisation des ventes par année
sales_by_year = merged_data.groupby('Year')['CA'].sum()
plt.figure(figsize=(10, 6))
sales_by_year.plot(kind='bar', color='skyblue')
plt.title('Ventes par année')
plt.xlabel('Année')
plt.ylabel('Ventes')
plt.xticks(rotation=45)
plt.show()

# Préparation des données pour la prédiction
X = merged_data[['Year']]
y = merged_data['CA']

# Division des données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle de régression linéaire
model = LinearRegression()

# Entraînement du modèle sur les données d'entraînement
model.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred = model.predict(X_test)

# Visualisation des prédictions
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title('Prédictions de ventes')
plt.xlabel('Année')
plt.ylabel('CA')
plt.show()

# Prédiction pour les années à venir
future_years = range(max(merged_data['Year']) + 1, max(merged_data['Year']) + 6)
future_sales_pred = model.predict(pd.DataFrame(future_years, columns=['Year']))

# Affichage des prédictions pour les années à venir
for year, sales_pred in zip(future_years, future_sales_pred):
    print(f"Prédiction de ventes pour l'année {year}: {sales_pred}")

