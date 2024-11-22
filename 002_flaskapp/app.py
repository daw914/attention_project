from flask import Flask, jsonify, request, render_template, redirect, url_for, session
import numpy as np
import pickle
import pandas as pd
import shap
import io
from trs import DataTransformer

# Load the pre-trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Collect form inputs
        gender = request.form["gender"]
        age = request.form["age"]
        education = request.form["education"]
        income = request.form["income"]
        marital_status = request.form["marital_status"]
        employment = request.form["employment"]
        cost = request.form["cost"]
        duration = request.form["duration"]
        firstclick = request.form["firstclick"]
        lastclick = request.form["lastclick"]
        clickcount = request.form["clickcount"]
        hispanic = request.form["hispanic"]
        black = request.form["black"]
        asian = request.form["asian"]
        Native_Hawaiian = request.form["Native_Hawaiian"]
        other = request.form["other"]
        position_1 = request.form["position_1"]
        position_2 = request.form["position_2"]
        position_3 = request.form["position_3"]
        position_4 = request.form["position_4"]
        position_5 = request.form["position_5"]
        order_l1 = request.form["order_l1"]
        order_l2 = request.form["order_l2"]
        order_l3 = request.form["order_l3"]
        order_l4 = request.form["order_l4"]
        order_l5 = request.form["order_l5"]
        

        # Prepare the input for the model
        input_data = np.array([[float(gender), float(age), float(education), float(income), float(marital_status), 
                                float(employment), float(cost), float(duration), float(firstclick),
                                float(lastclick), float(clickcount), float(hispanic), float(black),
                                float(asian), float(Native_Hawaiian), float(other), float(position_1),
                                float(position_2), float(position_3),float(position_4), float(position_5), float(order_l1),
                                float(order_l2), float(order_l3),float(order_l4), float(order_l5)]])

        # Convert input to DataFrame
        features = ['gender', 'age', 'education', 'income', 'marital_status', 'employment', 'cost', 'duration', 'firstclick', 'lastclick', 'clickcount', 'hispanic', 'black', 'asian', 'Native_Hawaiian', 'other', 'position_1', 'position_2', 'position_3', 'position_4', 'position_5', 'order_l1', 'order_l2', 'order_l3', 'order_l4', 'order_l5']
        X = pd.DataFrame(input_data, columns=features)

        # Make prediction
        pred = model.predict_proba(X)[0][1]
        print(f"Prediction: {pred}")  # Debugging output

        # Get the tree-based model (e.g., the model pipeline’s 'cat' step)
        tree_model = model.named_steps['xgb']

        # Initialize SHAP TreeExplainer
        explainer = shap.TreeExplainer(tree_model)

        # Transform input features as per your pipeline's transformation step
        X_shap = model.named_steps['tf'].transform(X)

        if isinstance(X_shap, pd.DataFrame):
            X_shap = X_shap.to_numpy()

        # Get SHAP values for the first instance
        shap_values = explainer.shap_values(X_shap)

        # Generate SHAP force plot
        shap.initjs()
        force_plot = shap.force_plot(
            explainer.expected_value, 
            shap_values[0], 
            X_shap[0],
            link='logit',
            feature_names=features,
        )

        # Save SHAP plot to an HTML string
        buf = io.StringIO()
        shap.save_html(buf, force_plot)
        shap_html = buf.getvalue()

        # Debug output to check if SHAP HTML is generated
        print(f"SHAP HTML: {shap_html[:500]}...")  # Print first 500 chars

        # Instead of redirecting, pass the data directly to render the result page
        return render_template("result.html", pred=pred, shap_html=shap_html)

    return render_template("index.html")


@app.route('/result')
def result():
    # Render the result page (without session)
    return "This should not be accessed directly."


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)