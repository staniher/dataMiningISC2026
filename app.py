from flask import Flask, render_template, request
import joblib
app=Flask(__name__)
model=joblib.load("kmeansModel.pkl")
sc=joblib.load("scaler.pkl")
def interpretationCluster(cluster_id):
	interpretation={
		0 :"Client intermediaire",
 		1:"Client premium à fort pouvoir d'achat et forte consommation",
		2:"Client impulsif à faible revenu",
		3:"Client intermediaire",
		4:"Client aisé mais peu depensier",
		5:"Client à faible revenu et faible engagement"
	}
	return interpretation.get(cluster_id, "Profil non identifié")

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/prediction", methods=["POST"])
def prediction():
	age=float(request.form['age'])
	revenu=float(request.form['revenu'])
	depense=float(request.form['depense'])
	import numpy as np 
	X=np.array([[age, revenu,depense]])
	X_transformes=sc.transform(X)
	cluster=int(model.predict(X_transformes)[0])
	profil=interpretationCluster(cluster)
	return render_template(
		"resultats.html", cluster=cluster, profil=profil,
		age=age, revenu=revenu,depense=depense
		)
if __name__=="__main__":
	app.run(debug=False)
