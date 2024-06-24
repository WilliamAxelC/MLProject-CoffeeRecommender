from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import surprise

app = Flask(__name__)
CORS(app)

# Load the trained model
algo = joblib.load('coffee_recommendation_svd_model.pkl')

# Load the dataset to map product IDs to details
df = pd.read_csv('../data/data.csv', delimiter=';')

# Load the model accuracy
with open('model_accuracy.txt', 'r') as f:
    model_accuracy = float(f.read())

# Helper function to predict ratings
def predict_rating(user_id, item_id):
    return algo.predict(user_id, item_id).est

def get_recommendations(store_location, product_category, product_type):
    # Encode the store_location using its integer value
    store_id = df[df['store_location'] == store_location]['store_id'].iloc[0]

    # Filter dataset based on user preferences
    filtered_df = df[(df['product_category'] == product_category) & 
                     (df['product_type'] == product_type)].drop_duplicates(subset=['product_id'])
    
    if filtered_df.empty:
        # Relax the filtering if no exact match is found
        filtered_df = df[(df['store_location'] == store_location) & 
                         (df['product_category'] == product_category)].drop_duplicates(subset=['product_id'])
    
    if filtered_df.empty:
        # Further relax the filtering if no matches are found
        filtered_df = df[df['store_location'] == store_location].drop_duplicates(subset=['product_id'])
    
    if filtered_df.empty:
        # If still no matches, use the entire dataset without filtering
        filtered_df = df.drop_duplicates(subset=['product_id'])
    
    # Get unique product IDs from the filtered dataframe
    unique_product_ids = filtered_df['product_id'].unique()
    
    # Predict ratings for each product ID
    recommendations = []
    seen_products = set()
    for product_id in unique_product_ids:
        if product_id not in seen_products:
            pred_rating = predict_rating(store_id, product_id)
            recommendations.append((product_id, pred_rating))
            seen_products.add(product_id)
    
    # Sort recommendations by predicted rating
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    
    # Get the top 5 recommendations
    top_recommendations = recommendations[:5]
    
    # Map product IDs to details
    recommended_coffees = []
    for product_id, _ in top_recommendations:
        coffee_details = df[df['product_id'] == product_id][['product_category', 'product_type', 'product_detail']].iloc[0].to_dict()
        recommended_coffees.append(coffee_details)
    
    return recommended_coffees

@app.route('/recommend', methods=['POST'])
def recommend():
    user_preferences = request.json.get('user_preferences', {})
    
    store_location = user_preferences.get('store_location')
    product_category = user_preferences.get('product_category')
    product_type = user_preferences.get('product_type')
    
    recommended_coffees = get_recommendations(store_location, product_category, product_type)
    
    return jsonify(recommended_coffees)

@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    return jsonify({'rmse': model_accuracy})

@app.route('/unique-store-locations', methods=['GET'])
def unique_store_locations():
    store_locations = df['store_location'].unique().tolist()
    return jsonify(store_locations)

@app.route('/unique-product-categories', methods=['GET'])
def unique_product_categories():
    product_categories = df['product_category'].unique().tolist()
    return jsonify(product_categories)

@app.route('/unique-product-types', methods=['GET'])
def unique_product_types():
    product_types = df['product_type'].unique().tolist()
    return jsonify(product_types)

if __name__ == '__main__':
    app.run(debug=True)
