import streamlit as st
import requests
import json
import re
from PIL import Image
import os
from google.cloud import vision

# === Configuration for Mistral API ===
MISTRAL_API_KEY = "YOUR_MISTRAL_API_KEY"  # Replace with your actual Mistral API key
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# === API Call Functions ===
def generate_meal_plan(profile):
    """
    Generate a personalized meal plan using Mistral AI API.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }
    
    # Build prompt based on profile, requesting structured days
    prompt = (
        f"Generate a personalized 7-day meal plan for a {profile['age']} year old {profile['gender']} "
        f"with weight {profile['weight']}kg, height {profile['height']}cm, activity level {profile['activity']}, "
        f"dietary preferences {', '.join(profile['dietary'])}, "
    )
    
    # Add menstrual cycle information only for females
    if profile['gender'] == "Female" and profile['menstrual_cycle'] != "Not Applicable":
        prompt += f"menstrual cycle phase {profile['menstrual_cycle']}, "
    
    prompt += (
        f"and a fitness goal of {profile['fitness_goal']}. "
        "Include macronutrient and micronutrient breakdown, hydration tips, and balanced meals for the day. "
        "Format each day clearly with 'DAY X: DAY_NAME' as a header (e.g., 'DAY 1: MONDAY'). "
        "For each day include sections for Breakfast, Lunch, Dinner, and Snacks."
    )
    
    payload = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "system", "content": "You are a helpful nutrition expert. Create structured meal plans with clear day headers."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1500  # Increased for a full week meal plan
    }
    
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        meal_plan = result["choices"][0]["message"]["content"]
    except Exception as e:
        meal_plan = f"Error generating meal plan: {e}"
    
    return meal_plan

def parse_meal_plan_by_day(meal_plan_text):
    """
    Parse the meal plan text to extract daily meal plans.
    Returns a dictionary with days as keys and meal content as values.
    """
    # Dictionary to store the meal plans for each day
    daily_plans = {}
    
    # Regular expression to find day headers (DAY X: DAY_NAME)
    day_pattern = r'DAY\s+\d+\s*:\s*([A-Z]+)'
    
    # Find all day headers
    day_matches = list(re.finditer(day_pattern, meal_plan_text, re.IGNORECASE))
    
    # Extract content for each day
    for i, match in enumerate(day_matches):
        day_name = match.group(1).capitalize()
        start_pos = match.start()
        
        # If this is not the last day, the end position is the start of the next day
        if i < len(day_matches) - 1:
            end_pos = day_matches[i + 1].start()
        else:
            end_pos = len(meal_plan_text)
        
        # Extract the text for this day
        day_content = meal_plan_text[start_pos:end_pos].strip()
        daily_plans[day_name] = day_content
    
    # If parsing fails, create a fallback
    if not daily_plans:
        daily_plans = {
            "Full Plan": meal_plan_text
        }
    
    return daily_plans

def recognize_food(image_bytes):
    """
    Recognize food items in an image using Google Cloud Vision API.
    """
    try:
        # Create a client for the Vision API
        client = vision.ImageAnnotatorClient()
        
        # Create an image instance
        image = vision.Image(content=image_bytes)
        
        # Perform label detection on the image
        response = client.label_detection(image=image)
        labels = response.label_annotations
        
        # Filter for food-related labels
        food_labels = []
        food_keywords = ["food", "dish", "meal", "cuisine", "vegetable", "fruit", "meat", "dessert", 
                        "breakfast", "lunch", "dinner", "snack", "ingredient", "recipe", "plate"]
        
        for label in labels:
            # Check if the label is food-related
            is_food = any(keyword in label.description.lower() for keyword in food_keywords) or label.score > 0.7
            if is_food:
                food_labels.append({
                    "description": label.description,
                    "score": f"{label.score:.2f}"
                })
        
        # If no food labels are detected
        if not food_labels:
            return "No food items detected in this image. Try uploading a clearer image of your meal."
        
        # Get nutritional information using Mistral API for the detected food items
        food_items = [label["description"] for label in food_labels]
        nutritional_info = get_food_nutrition(food_items)
        
        # Combine results
        result = {
            "detected_items": food_labels,
            "nutritional_info": nutritional_info
        }
        
        return result
        
    except Exception as e:
        return f"Error recognizing food: {str(e)}"

def get_food_nutrition(food_items):
    """
    Get nutritional information for detected food items using Mistral API.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }
    
    # Build prompt for nutrition information
    food_list = ", ".join(food_items)
    prompt = (
        f"As a nutrition expert, please provide a short nutritional analysis for these food items: {food_list}. "
        "Include estimated calories, macronutrients (protein, carbs, fat), and key micronutrients where applicable. "
        "Keep the analysis concise and informative."
    )
    
    payload = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "system", "content": "You are a nutrition expert providing concise food analysis."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 300
    }
    
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error getting nutritional info: {str(e)}"

# === Streamlit UI ===
def main():
    st.title("🍽 AI-Powered Meal Plan Generator")
    
    # Check for Google Cloud credentials
    if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
        st.sidebar.warning("Google Cloud credentials not set. Food recognition may not work properly.")
        st.sidebar.info(
            "To set up credentials, create a service account key and set the environment variable:\n\n"
            "```\nos.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/denizasuleymanova/python_3/FOODBOT3000/foodapp-453613-16f8a603a026.json'\n```\n\n"
            "You can add this at the top of your script or use a .env file."
        )
    
    # Initialize session state for storing the meal plan
    if 'meal_plan' not in st.session_state:
        st.session_state.meal_plan = None
    if 'daily_plans' not in st.session_state:
        st.session_state.daily_plans = {}
    if 'current_day' not in st.session_state:
        st.session_state.current_day = None
    if 'food_log' not in st.session_state:
        st.session_state.food_log = []
    
    # User Profile Setup
    st.sidebar.header("User Profile Setup")
    age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30)
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    activity = st.sidebar.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Active", "Very Active"])
    dietary = st.sidebar.multiselect("Dietary Preferences", ["Vegan", "Vegetarian", "Keto", "Halal", "Gluten-Free", "None"])
    
    # Show menstrual cycle only for females
    menstrual_cycle = "Not Applicable"
    if gender == "Female":
        menstrual_cycle = st.sidebar.selectbox("Menstrual Cycle Phase", ["Not Applicable", "Follicular", "Ovulatory", "Luteal", "Menstrual"])
    
    fitness_goal = st.sidebar.selectbox("Fitness Goals", ["Weight Loss", "Muscle Gain", "Maintenance"])
    
    profile = {
        "age": age,
        "gender": gender,
        "weight": weight,
        "height": height,
        "activity": activity,
        "dietary": dietary if dietary else ["None"],
        "menstrual_cycle": menstrual_cycle,
        "fitness_goal": fitness_goal,
    }
    
    # Meal Plan Generation Section
    st.header("Generate Your Meal Plan")
    if st.button("Generate 7-Day Meal Plan"):
        with st.spinner("Generating meal plan..."):
            meal_plan = generate_meal_plan(profile)
            st.session_state.meal_plan = meal_plan
            
            # Parse the meal plan by day
            daily_plans = parse_meal_plan_by_day(meal_plan)
            st.session_state.daily_plans = daily_plans
            
            # Set current day to the first day in the plan
            if daily_plans:
                st.session_state.current_day = list(daily_plans.keys())[0]
    
    # Display meal plan with day selection buttons
    if st.session_state.meal_plan:
        # Create buttons for days of the week
        st.subheader("Navigate Your Meal Plan")
        cols = st.columns(len(st.session_state.daily_plans))
        
        for i, day in enumerate(st.session_state.daily_plans.keys()):
            if cols[i].button(day):
                st.session_state.current_day = day
        
        # Display the selected day's meal plan
        if st.session_state.current_day:
            st.markdown(f"### {st.session_state.current_day}'s Meal Plan")
            st.markdown(st.session_state.daily_plans[st.session_state.current_day])
        
        # Option to view the complete meal plan
        with st.expander("View Complete Meal Plan"):
            st.markdown("### Your Complete 7-Day Meal Plan")
            st.write(st.session_state.meal_plan)
    
#    # Food Recognition Section
#    st.header("Food Recognition & Logging")
#    uploaded_file = st.file_uploader("Upload an image of your meal", type=["jpg", "jpeg", "png"])
#    
#    if uploaded_file is not None:
#        image = Image.open(uploaded_file)
#        st.image(image, caption="Uploaded Meal", use_column_width=True)
#        
#        if st.button("Recognize Food"):
#            with st.spinner("Analyzing your food with Google Cloud Vision API..."):
#                food_info = recognize_food(uploaded_file.getvalue())
#                
#                st.markdown("### Food Analysis Results")
#                
#                if isinstance(food_info, dict):
#                    # Display detected food items
#                    st.subheader("Detected Food Items")
#                    for item in food_info["detected_items"]:
#                        st.write(f"• {item['description']} (Confidence: {item['score']})")
#                    
#                    # Display nutritional information
#                    st.subheader("Nutritional Information")
#                    st.write(food_info["nutritional_info"])
#                    
#                    # Add to food log
#                    log_entry = {
#                       "day": st.session_state.current_day if st.session_state.current_day else "Unknown",
#                        "time": "Current time",
#                        "items": [item["description"] for item in food_info["detected_items"]],
#                        "nutrition": food_info["nutritional_info"]
#                    }
#                    
#                    st.session_state.food_log.append(log_entry)
#                else:
#                    st.write(food_info)  # Display error message
    
    # Meal Plan History
    st.header("Food Log")
    if st.session_state.food_log:
        for i, entry in enumerate(st.session_state.food_log):
            with st.expander(f"Meal {i+1} - {entry['day']}"):
                st.write(f"**Items detected:** {', '.join(entry['items'])}")
                st.write("**Nutritional Analysis:**")
                st.write(entry['nutrition'])
    else:
        st.write("No food logged yet. Use the Food Recognition feature to start logging your meals.")

if __name__ == "__main__":
    main()