"""
Recipe Recommender Web App
Beautiful interface for chefs to discover similar recipes
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import os

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Recipe Recommender",
    page_icon="👨‍🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Title */
    .title {
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 28px;
        border: none;
        font-size: 16px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Cards */
    .recipe-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s;
    }
    
    .recipe-card:hover {
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        transform: translateX(5px);
    }
    
    /* Badges */
    .badge-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .badge-medium {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .badge-low {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    
    /* Stats */
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    
    .stat-number {
        font-size: 2.5em;
        font-weight: bold;
    }
    
    .stat-label {
        font-size: 0.9em;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource(show_spinner=False)
def load_model():
    import os
    import pickle
    from huggingface_hub import hf_hub_download
    import gc

    # ✅ FIX: Use a writable cache directory for Streamlit Cloud
    cache_dir = os.path.join(os.getcwd(), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    model_file = hf_hub_download(
        repo_id="anahitmanukyan/recipe_dataset",
        filename="recipe_recommender_large.pkl",
        repo_type="dataset",
        cache_dir=cache_dir  # ✅ FIX: Specify writable cache location
    )

    st.success("✅ Model loaded")

    with open(model_file, "rb") as f:
        data = pickle.load(f)

    # Explicit cleanup (important for Streamlit Cloud)
    gc.collect()

    return data["df"], data["tfidf"], data["tfidf_matrix"]

# Load data
df, tfidf, tfidf_matrix = load_model()

# ============================================
# HELPER FUNCTIONS
# ============================================

def search_recipes(query, n=50):
    """Search for recipes by name or ingredient"""
    query = query.lower()
    mask = (
        df['title'].str.lower().str.contains(query, na=False, regex=False) |
        df['ingredients'].str.lower().str.contains(query, na=False, regex=False)
    )
    results = df[mask].head(n)
    return results

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_recommendations(recipe_name, n=10, min_similarity=0.1):
    try:
        idx = df[df['title'].str.lower() == recipe_name.lower()].index[0]

        query_vec = tfidf_matrix[idx]

        # Compute similarities (sparse-safe)
        similarities = cosine_similarity(query_vec, tfidf_matrix).ravel()

        # Get top K candidates WITHOUT full sort
        top_k = np.argpartition(similarities, -50)[-50:]

        # Filter + sort only top candidates
        filtered = [
            i for i in top_k
            if i != idx and similarities[i] >= min_similarity
        ]

        filtered = sorted(filtered, key=lambda i: similarities[i], reverse=True)[:n]

        results = df.iloc[filtered][['title', 'ingredients', 'directions']].copy()
        results['similarity_score'] = similarities[filtered]

        return results

    except IndexError:
        return None

def recommend_by_ingredients(ingredients_list, n=15, min_similarity=0.05):
    """Find recipes based on ingredients you have (cloud-safe)"""

    user_text = ' '.join(ingredients_list).lower()
    user_vec = tfidf.transform([user_text])

    # Compute similarities (1 × N, sparse-friendly)
    similarities = cosine_similarity(user_vec, tfidf_matrix).ravel()

    # Take only top-K candidates instead of sorting everything
    top_k = np.argpartition(similarities, -100)[-100:]

    # Filter + rank
    filtered = [
        i for i in top_k
        if similarities[i] >= min_similarity
    ]

    filtered = sorted(filtered, key=lambda i: similarities[i], reverse=True)[:n]

    results = df.iloc[filtered][['title', 'ingredients', 'directions']].copy()
    results['match_score'] = similarities[filtered]

    return results

def get_similarity_badge(score):
    """Return HTML badge based on similarity score"""
    if score >= 0.7:
        return '<span class="badge-high">🔥 Highly Similar</span>'
    elif score >= 0.5:
        return '<span class="badge-medium">⭐ Good Match</span>'
    else:
        return '<span class="badge-low">✓ Similar</span>'

# ============================================
# HEADER
# ============================================

st.markdown('<div class="title">👨‍🍳 Chef\'s Recipe Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover similar recipes and find culinary inspiration</div>', unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.image("https://em-content.zobj.net/source/animated-noto-color-emoji/356/cook_1f9d1-200d-1f373.gif", width=150)
    
    st.markdown("---")
    
    # Stats
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-number">{len(df):,}</div>
        <div class="stat-label">Recipes Available</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 How to Use")
    st.markdown("""
    **Find Similar Recipes:**
    1. Search for a recipe you like
    2. Select it from the results
    3. Get personalized recommendations
    
    **What Can I Make?**
    - Enter ingredients you have
    - Discover recipes you can cook
    - Perfect for using up ingredients!
    """)
    
    st.markdown("---")
    
    st.info("📝 **Note:** This dataset includes user-submitted recipes from home cooks. Spelling and grammar may vary - part of the charm! 😄")
    
    st.markdown("---")
    
    
    st.markdown("### ⚙️ Settings")
    num_recommendations = st.slider(
        "Number of recommendations:",
        min_value=5,
        max_value=30,
        value=10,
        help="How many similar recipes to show"
    )
    
    min_similarity = st.slider(
        "Minimum similarity:",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Filter out recipes that are too different"
    )

# ============================================
# MAIN CONTENT - TABS
# ============================================

tab1, tab2, tab3 = st.tabs(["🔍 Find Similar Recipes", "🥘 What Can I Make?", "📊 Random Discovery"])

# ============================================
# TAB 1: FIND SIMILAR RECIPES
# ============================================

with tab1:
    st.markdown("### Search for a Recipe")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search by recipe name or ingredient:",
            placeholder="e.g., chocolate cake, chicken pasta, garlic bread...",
            key="search_input"
        )
    
    with col2:
        search_button = st.button("🔍 Search", key="search_btn", use_container_width=True)
    
    if search_query or search_button:
        if search_query:
            with st.spinner("Searching recipes..."):
                search_results = search_recipes(search_query, n=50)
            
            if len(search_results) > 0:
                st.success(f"✅ Found {len(search_results)} recipes!")
                
                # Recipe selection
                st.markdown("### Select a Recipe")
                selected_recipe = st.selectbox(
                    "Choose a recipe to get recommendations:",
                    search_results['title'].tolist(),
                    key="recipe_select"
                )
                
                # Show selected recipe details
                selected_data = search_results[search_results['title'] == selected_recipe].iloc[0]
                
                with st.expander("📖 View Recipe Details", expanded=False):
                    st.markdown(f"**Ingredients:**")
                    st.write(selected_data['ingredients'])
                    st.markdown(f"**Directions:**")
                    st.write(selected_data['directions'])
                
                # Get recommendations button
                if st.button("✨ Get Similar Recipes", key="rec_btn", use_container_width=True):
                    with st.spinner("Finding similar recipes..."):
                        recommendations = get_recommendations(
                            selected_recipe, 
                            n=num_recommendations,
                            min_similarity=min_similarity
                        )
                    
                    if recommendations is not None and len(recommendations) > 0:
                        st.markdown("---")
                        st.markdown(f"### ✨ Recipes Similar to: *{selected_recipe}*")
                        st.markdown(f"<p style='color: #7f8c8d;'>Found {len(recommendations)} similar recipes</p>", unsafe_allow_html=True)
                        
                        # Display recommendations
                        for idx, row in recommendations.iterrows():
                            similarity = row['similarity_score']
                            
                            # Create expandable card for each recommendation
                            with st.expander(f"**{row['title']}** - {similarity:.1%} match"):
                                # Badge
                                st.markdown(get_similarity_badge(similarity), unsafe_allow_html=True)
                                
                                col1, col2 = st.columns([1, 1])
                                
                                with col1:
                                    st.markdown("**🥘 Ingredients:**")
                                    st.write(row['ingredients'])
                                
                                with col2:
                                    st.markdown("**📝 Directions:**")
                                    st.write(row['directions'])
                    else:
                        st.warning("No similar recipes found. Try adjusting the minimum similarity setting.")
            else:
                st.warning("❌ No recipes found. Try different search terms!")
        else:
            st.info("👆 Enter a search term to get started")

# ============================================
# TAB 2: WHAT CAN I MAKE?
# ============================================

with tab2:
    st.markdown("### What's in Your Kitchen?")
    st.markdown("Enter ingredients you have on hand and discover recipes you can make!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ingredients_input = st.text_area(
            "Your ingredients (one per line):",
            height=200,
            placeholder="chicken breast\ngarlic\ntomatoes\nolive oil\nbasil\nonion\nsalt\npepper",
            key="ingredients_input"
        )
    
    with col2:
        st.markdown("**💡 Tips:**")
        st.markdown("""
        - Enter one ingredient per line
        - Be specific: "chicken breast" not just "chicken"
        - Include spices and seasonings
        - More ingredients = better matches
        """)
        
        num_results = st.slider(
            "Number of recipes:",
            min_value=5,
            max_value=30,
            value=15,
            key="ingredient_slider"
        )
    
    if st.button("🍳 Find Recipes", key="find_recipes_btn", use_container_width=True):
        if ingredients_input.strip():
            ingredients_list = [ing.strip().lower() for ing in ingredients_input.split('\n') if ing.strip()]
            
            if ingredients_list:
                with st.spinner("Searching for recipes..."):
                    results = recommend_by_ingredients(ingredients_list, n=num_results)
                
                st.markdown("---")
                st.success(f"✅ Found {len(results)} recipes you can make!")
                st.markdown(f"<p style='color: #7f8c8d;'>Based on {len(ingredients_list)} ingredients</p>", unsafe_allow_html=True)
                
                # Display results
                for idx, row in results.iterrows():
                    match_score = row['match_score']
                    match_pct = match_score * 100
                    
                    with st.expander(f"**{row['title']}** - {match_pct:.0f}% match"):
                        # Progress bar
                        st.progress(min(match_score, 1.0))
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("**🥘 All Ingredients Needed:**")
                            st.write(row['ingredients'])
                        
                        with col2:
                            st.markdown("**📝 Instructions:**")
                            st.write(row['directions'])
                        
                        # Show which ingredients match
                        recipe_ingredients_lower = row['ingredients'].lower()
                        matching = [ing for ing in ingredients_list if ing in recipe_ingredients_lower]
                        
                        if matching:
                            st.markdown(f"**✓ You have:** {', '.join(matching)}")
            else:
                st.warning("Please enter at least one ingredient!")
        else:
            st.info("👆 Enter your ingredients to get started")

# ============================================
# TAB 3: RANDOM DISCOVERY
# ============================================

with tab3:
    st.markdown("### 🎲 Discover Something New!")
    st.markdown("Feeling adventurous? Get random recipe recommendations!")
    
    # Category buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🍰 Random Dessert", use_container_width=True, key="btn_dessert"):
            with st.spinner("Finding a random dessert..."):
                dessert_recipes = df[df['ingredients'].str.contains('sugar|chocolate|cake|cookie|cream|vanilla', case=False, na=False)]
                if len(dessert_recipes) > 0:
                    random_recipe = dessert_recipes.sample(1).iloc[0]
                    st.session_state['random_recipe'] = random_recipe['title']
                    st.session_state['random_category'] = 'Dessert'
                    st.rerun()
    
    with col2:
        if st.button("🍗 Random Main Course", use_container_width=True, key="btn_main"):
            with st.spinner("Finding a random main course..."):
                main_recipes = df[df['ingredients'].str.contains('chicken|beef|pork|fish|lamb|turkey', case=False, na=False)]
                if len(main_recipes) > 0:
                    random_recipe = main_recipes.sample(1).iloc[0]
                    st.session_state['random_recipe'] = random_recipe['title']
                    st.session_state['random_category'] = 'Main Course'
                    st.rerun()
    
    with col3:
        if st.button("🥗 Random Side Dish", use_container_width=True, key="btn_side"):
            with st.spinner("Finding a random side dish..."):
                side_recipes = df[df['ingredients'].str.contains('potato|rice|salad|vegetable|beans|corn', case=False, na=False)]
                if len(side_recipes) > 0:
                    random_recipe = side_recipes.sample(1).iloc[0]
                    st.session_state['random_recipe'] = random_recipe['title']
                    st.session_state['random_category'] = 'Side Dish'
                    st.rerun()
    
    with col4:
        if st.button("🎲 Surprise Me!", use_container_width=True, key="btn_surprise"):
            with st.spinner("Finding a random recipe..."):
                random_recipe = df.sample(1).iloc[0]
                st.session_state['random_recipe'] = random_recipe['title']
                st.session_state['random_category'] = 'Random'
                st.rerun()
    
    # Display selected random recipe
    if 'random_recipe' in st.session_state:
        st.markdown("---")
        
        recipe_name = st.session_state['random_recipe']
        category = st.session_state.get('random_category', 'Random')
        
        # Header with category badge
        st.markdown(f"### 🎯 {category}: *{recipe_name}*")
        
        # Get recipe data
        recipe_data = df[df['title'] == recipe_name].iloc[0]
        
        # Show recipe details
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**🥘 Ingredients:**")
            st.write(recipe_data['ingredients'])
        
        with col2:
            st.markdown("**📝 Directions:**")
            st.write(recipe_data['directions'])
        
        st.markdown("---")
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✨ Get Similar Recipes", key="random_rec_btn", use_container_width=True):
                st.session_state['show_random_recommendations'] = True
                st.rerun()
        
        with col2:
            if st.button("🔄 Try Another Recipe", key="try_another_btn", use_container_width=True):
                # Get another random recipe from the same category
                if category == 'Dessert':
                    dessert_recipes = df[df['ingredients'].str.contains('sugar|chocolate|cake|cookie|cream|vanilla', case=False, na=False)]
                    random_recipe = dessert_recipes.sample(1).iloc[0]
                elif category == 'Main Course':
                    main_recipes = df[df['ingredients'].str.contains('chicken|beef|pork|fish|lamb|turkey', case=False, na=False)]
                    random_recipe = main_recipes.sample(1).iloc[0]
                elif category == 'Side Dish':
                    side_recipes = df[df['ingredients'].str.contains('potato|rice|salad|vegetable|beans|corn', case=False, na=False)]
                    random_recipe = side_recipes.sample(1).iloc[0]
                else:
                    random_recipe = df.sample(1).iloc[0]
                
                st.session_state['random_recipe'] = random_recipe['title']
                st.session_state['show_random_recommendations'] = False
                st.rerun()
        
        # Show recommendations if requested
        if st.session_state.get('show_random_recommendations', False):
            with st.spinner("Finding similar recipes..."):
                recommendations = get_recommendations(recipe_name, n=10)
            
            if recommendations is not None and len(recommendations) > 0:
                st.markdown("---")
                st.markdown("### ✨ Similar Recipes:")
                
                for idx, row in recommendations.iterrows():
                    with st.expander(f"**{row['title']}** - {row['similarity_score']:.1%} match"):
                        st.markdown(get_similarity_badge(row['similarity_score']), unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("**Ingredients:**")
                            st.write(row['ingredients'])
                        
                        with col2:
                            st.markdown("**Directions:**")
                            st.write(row['directions'])
    else:
        # Instructions when nothing is selected
        st.info("👆 Click any button above to discover a random recipe!")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p>Made with ❤️ for chefs by chefs</p>
    <p style='font-size: 0.8em;'>Powered by TF-IDF similarity matching • {num_recipes:,} recipes • Built with Streamlit</p>
</div>
""".format(num_recipes=len(df)), unsafe_allow_html=True)
