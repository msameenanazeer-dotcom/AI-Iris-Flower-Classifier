from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

iris = load_iris()

model = DecisionTreeClassifier()
model.fit(iris.data, iris.target)

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":
        values = [
            float(request.form["sepal_length"]),
            float(request.form["sepal_width"]),
            float(request.form["petal_length"]),
            float(request.form["petal_width"])
        ]

        prediction = model.predict([values])[0]
        result = iris.target_names[prediction]

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
