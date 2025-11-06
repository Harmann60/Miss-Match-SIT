import pandas as pd
import io
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
import threading

# --- [ Part 1: Global Variables ] ---
# We create global variables to hold our data and model.
# This way, we only load and process them ONCE when the server starts.
app = Flask(__name__)
CORS(app)  # Allows the Android app to talk to this server

# We'll store our key objects here
df_clean = None
profile_matrix = None
preprocessor = None
data_is_loaded = False


# --- [ Part 2: Data Cleaning Code (from your first cell) ] ---
def load_and_clean_data(file_name='profiles.csv'):
    """
    This function loads and cleans the raw CSV data.
    It combines all the logic from your first Colab cell.
    """
    global df_clean, preprocessor, profile_matrix # We're modifying the global variables

    print("--- [Server] Step 1: Loading Data ---")
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"\nERROR: '{file_name}' not found.")
        print("Please make sure 'profiles.csv' is in the same folder as 'app.py'")
        return

    print("--- [Server] Step 2: Filtering for Target Audience ---")
    df = df[
        (df['sex'] == 'm') & (df['orientation'] == 'straight') |
        (df['sex'] == 'f') & (df['orientation'] == 'straight')
    ]

    print("--- [Server] Step 3: Selecting Useful Columns ---")
    essay_cols = [
        'essay0', 'essay1', 'essay2', 'essay3', 'essay4',
        'essay5', 'essay6', 'essay7', 'essay8', 'essay9'
    ]
    profile_cols = [
        'age', 'sex', 'orientation', 'body_type', 'diet', 'drinks',
        'drugs', 'education', 'job', 'religion', 'status'
    ]
    all_needed_cols = profile_cols + essay_cols
    existing_cols = [col for col in all_needed_cols if col in df.columns]
    df_clean = df[existing_cols].copy()

    print("--- [Server] Step 4: Cleaning Missing Values ---")
    categorical_cols = [
        'body_type', 'diet', 'drinks', 'drugs', 'education', 'job', 'religion', 'status'
    ]
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna('unknown')
        
    existing_essay_cols = [col for col in essay_cols if col in df_clean.columns]
    df_clean[existing_essay_cols] = df_clean[existing_essay_cols].fillna('')

    print("--- [Server] Step 5: Combining Essays into One 'Bio' ---")
    df_clean['bio'] = df_clean[existing_essay_cols].apply(lambda x: ' '.join(x), axis=1)

    print("--- [Server] Step 6: Finalizing the Clean DataFrame ---")
    df_clean = df_clean.drop(columns=existing_essay_cols)
    df_clean = df_clean.reset_index(drop=True) # CRITICAL: Reset the index
    print("Clean data is ready.")


    # --- [ Part 3: Model Building Code (from your second cell) ] ---
    print("--- [Server] Step 7: Preparing Data for Modeling ---")
    categorical_cols_model = [
        'sex', 'orientation', 'body_type', 'diet', 'drinks',
        'drugs', 'education', 'job', 'religion', 'status'
    ]
    numerical_cols = ['age']
    text_col = 'bio'

    print("--- [Server] Step 8: Building the Vectorizer Pipeline ---")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    min_max_scaler = MinMaxScaler()

    # We use our global preprocessor variable
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', tfidf_vectorizer, text_col),
            ('categorical', one_hot_encoder, categorical_cols_model),
            ('numerical', min_max_scaler, numerical_cols)
        ],
        remainder='drop'
    )

    print("--- [Server] Step 9: Fitting the Pipeline ---")
    # This is the "training" step. It creates the big matrix.
    profile_matrix = preprocessor.fit_transform(df_clean)

    print("--- [Server] Step 10: MODEL IS READY! ---")
    print(f"Shape of the matrix: {profile_matrix.shape}")
    print("Server is now ready to accept requests.")
    
    global data_is_loaded
    data_is_loaded = True


# --- [ Part 4: The "Get Matches" Function (from your second cell) ] ---
def get_matches(user_index, n=10):
    """
    Finds the top N most similar users for a given user.
    """
    global df_clean, profile_matrix # Use the loaded data
    
    # 1. Get the target user's vector
    user_vector = profile_matrix[user_index]
    
    # 2. Get the target user's 'sex'
    user_sex = df_clean.loc[user_index, 'sex']
    
    # 3. Create a "target matrix" of ONLY the opposite sex
    if user_sex == 'm':
        target_indices = df_clean[df_clean['sex'] == 'f'].index
    elif user_sex == 'f':
        target_indices = df_clean[df_clean['sex'] == 'm'].index
    else:
        return pd.DataFrame() 
        
    target_matrix = profile_matrix[target_indices]
    
    # 4. Calculate Cosine Similarity
    sim_scores = cosine_similarity(user_vector, target_matrix)
    
    # 5. Get the top N scores
    top_n_indices_in_target_matrix = np.argsort(sim_scores[0])[-n:][::-1]
    
    # 6. Map indices back to original
    top_n_original_indices = [target_indices[i] for i in top_n_indices_in_target_matrix]
    
    # 7. Get scores
    top_n_scores = sim_scores[0][top_n_indices_in_target_matrix]
    
    # 8. Get profiles
    top_n_profiles = df_clean.iloc[top_n_original_indices].copy()
    
    # 9. Add score
    top_n_profiles['similarity_score'] = top_n_scores
    
    return top_n_profiles


# --- [ Part 5: The API Endpoints (The "Bridge") ] ---

@app.route('/')
def home():
    """A simple 'hello' route to check if the server is running."""
    return "Dating App Matchmaking Server is ALIVE!"

@app.route('/get-matches-for-user/<int:user_index>')
def get_matches_api(user_index):
    """
    This is the main API endpoint your Android app will call.
    """
    global data_is_loaded
    if not data_is_loaded:
        return jsonify({"error": "Server is still loading. Please try again in a minute."}), 503

    print(f"\nRequest received: Find matches for user {user_index}")

    # Check if user_index is valid
    if user_index not in df_clean.index:
        return jsonify({"error": "User index out of bounds."}), 404

    # 1. Run your function
    matches_df = get_matches(user_index, n=10)
    
    # 2. Convert the DataFrame of matches to JSON
    # 'orient=records' makes it a nice list of objects
    matches_json = matches_df.to_json(orient="records")
    
    # 3. Send the JSON back to the Android app
    # We use jsonify to send it correctly
    return jsonify(matches_df.to_dict(orient="records"))

@app.route('/get-user-profile/<int:user_index>')
def get_user_profile_api(user_index):
    """
    A helpful endpoint to get the profile of a single user.
    """
    global data_is_loaded
    if not data_is_loaded:
        return jsonify({"error": "Server is still loading. Please try again in a minute."}), 503

    if user_index not in df_clean.index:
        return jsonify({"error": "User index out of bounds."}), 404
        
    user_profile = df_clean.iloc[[user_index]]
    return jsonify(user_profile.to_dict(orient="records"))


# --- [ Part 6: Run the Server ] ---
if __name__ == '__main__':
    # We use a thread to load the data *after* the server starts
    # This way, the server can respond "I'm loading" immediately
    print("Starting data loading process in a background thread...")
    threading.Thread(target=load_and_clean_data).start()
    
    # This runs the Flask app
    # host='0.0.0.0' means it's accessible from other devices on your network
    app.run(host='0.0.0.0', port=5000)
