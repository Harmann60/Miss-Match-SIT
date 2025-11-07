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
import firebase_admin
from firebase_admin import credentials, firestore

# --- [ Part 1: Global Variables & Firebase Setup ] ---
app = Flask(__name__)
CORS(app)  # Allows the Android app to talk to this server

# --- Firebase Setup ---
try:
    # Initialize Firebase
    # This looks for the private key file you downloaded
    cred = credentials.Certificate("serviceAccountKey.json") 
    firebase_admin.initialize_app(cred)
    db = firestore.client() # This is our connection to the database
    print("--- [Server] Successfully connected to Firebase ---")
except FileNotFoundError:
    print("--- [Server] WARNING: serviceAccountKey.json not found. ---")
    print("--- [Server] Match-making (swiping) will NOT work. ---")
    db = None
except Exception as e:
    print(f"--- [Server] Error initializing Firebase: {e} ---")
    db = None

# --- Model Variables ---
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
    # CRITICAL: We reset the index. The index (row number) will be our "User ID"
    df_clean = df_clean.reset_index(drop=True) 
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


# --- [ Part 4: The "Get Matches" Function (with user_id added) ] ---
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
    
    # --- THIS IS THE UPDATED LINE ---
    # We must add the user's ID (which is the index) to the data
    # so the frontend knows who to send in the "/swipe" request.
    top_n_profiles['user_id'] = top_n_profiles.index
    # --- END OF UPDATED LINE ---
    
    return top_n_profiles


# --- [ Part 5: The API Endpoints (NOW WITH /SWIPE!) ] ---

@app.route('/')
def home():
    """A simple 'hello' route to check if the server is running."""
    return "Dating App Matchmaking Server is ALIVE!"

@app.route('/get-matches-for-user/<int:user_index>')
def get_matches_api(user_index):
    """
    This is the main API endpoint your Android app will call
    to get the list of profiles to show in the swipe deck.
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
    return jsonify(matches_df.to_dict(orient="records"))

@app.route('/get-user-profile/<int:user_index>')
def get_user_profile_api(user_index):
    """
    A helpful endpoint to get the profile of a single user.
    We use the row number (index) as the User ID.
    """
    global data_is_loaded
    if not data_is_loaded:
        return jsonify({"error": "Server is still loading. Please try again in a minute."}), 503

    if user_index not in df_clean.index:
        return jsonify({"error": "User index out of bounds."}), 404
        
    user_profile = df_clean.iloc[[user_index]]
    return jsonify(user_profile.to_dict(orient="records"))


# --- [ THIS IS THE NEW FUNCTION! ] ---
@app.route('/swipe', methods=['POST'])
def handle_swipe():
    """
    This is the NEW endpoint for when a user swipes.
    The Android app will send a POST request here.
    """
    if not db:
        return jsonify({"error": "Database is not connected."}), 500

    # 1. Get the data from the Android app's request
    try:
        data = request.json
        # We convert user IDs to strings, as Firebase likes them
        swiper_id = str(data['swiper_id'])
        swiped_on_id = str(data['swiped_on_id'])
        action = data['action'] # "like" or "dislike"
    except:
        return jsonify({"error": "Invalid request. Missing swiper_id, swiped_on_id, or action."}), 400

    print(f"\nSwipe received: User {swiper_id} {action}d User {swiped_on_id}")

    # 2. If it's a "dislike", just record it and we're done.
    if action == 'dislike':
        dislike_data = {'action': 'dislike', 'timestamp': firestore.SERVER_TIMESTAMP}
        # We store dislikes in a 'swipes' collection
        db.collection('swipes').document(f"{swiper_id}_{swiped_on_id}").set(dislike_data)
        return jsonify({"match": False, "message": "Dislike recorded."})

    # 3. If it's a "like", we record it AND check for a mutual match
    if action == 'like':
        # Create the 'like' document in the database
        like_data = {'action': 'like', 'timestamp': firestore.SERVER_TIMESTAMP}
        my_like_ref = db.collection('swipes').document(f"{swiper_id}_{swiped_on_id}")
        my_like_ref.set(like_data)

        # Now, check if the *other person* has liked us back
        their_like_ref = db.collection('swipes').document(f"{swiped_on_id}_{swiper_id}")
        their_like_doc = their_like_ref.get()

        if their_like_doc.exists and their_like_doc.to_dict().get('action') == 'like':
            # --- IT'S A MUTUAL MATCH! ---
            print(f"*** MUTUAL MATCH: {swiper_id} and {swiped_on_id} ***")
            # Create a "match" document
            match_data = {
                'users': [swiper_id, swiped_on_id],
                'timestamp': firestore.SERVER_TIMESTAMP
            }
            db.collection('matches').add(match_data) # Add to the 'matches' collection
            
            # Tell the Android app "IT'S A MATCH!"
            return jsonify({"match": True})
        else:
            # --- Not a mutual match (yet) ---
            # They haven't liked us back, so we just tell the app "Like recorded".
            return jsonify({"match": False, "message": "Like recorded."})

    return jsonify({"error": "Invalid action."}), 400


# --- [ Part 6: Run the Server ] ---
if __name__ == '__main__':
    # We use a thread to load the data *after* the server starts
    print("Starting data loading process in a background thread...")
    threading.Thread(target=load_and_clean_data).start()
    
    # This runs the Flask app
    app.run(host='0.0.0.0', port=5000)
