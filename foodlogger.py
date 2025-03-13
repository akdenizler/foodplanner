import streamlit as st
import base64
import os
import requests
import dotenv
import json
import re
from PIL import Image
import matplotlib.pyplot as plt
from mistralai import Mistral

dotenv.load_dotenv()

# === Configuration for Mistral API ===
# For meal plan generation, we use the HTTP endpoint.
MEAL_PLAN_API_KEY = os.environ.get("MISTRAL_API_KEY")
MEAL_PLAN_API_URL = "https://api.mistral.ai/v1/chat/completions"
# For food recognition, we use the Mistral Python client.
FOOD_MODEL = "pixtral-12b-2409"

# ============================
# Functions for Meal Plan Generator
# ============================
def generate_meal_plan(profile):
    """
    Generate a personalized 7-day meal plan using Mistral AI API.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MEAL_PLAN_API_KEY}"
    }
    
    # Build prompt based on profile, requesting structured days.
    prompt = (
        f"Generate a personalized 7-day meal plan for a {profile['age']} year old {profile['gender']} "
        f"with weight {profile['weight']}kg, height {profile['height']}cm, activity level {profile['activity']}, "
        f"dietary preferences {', '.join(profile['dietary'])}, "
    )
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
        "max_tokens": 1500
    }
    
    try:
        response = requests.post(MEAL_PLAN_API_URL, headers=headers, json=payload)
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
    daily_plans = {}
    day_pattern = r'DAY\s+\d+\s*:\s*([A-Z]+)'
    day_matches = list(re.finditer(day_pattern, meal_plan_text, re.IGNORECASE))
    
    for i, match in enumerate(day_matches):
        day_name = match.group(1).capitalize()
        start_pos = match.start()
        end_pos = day_matches[i + 1].start() if i < len(day_matches) - 1 else len(meal_plan_text)
        day_content = meal_plan_text[start_pos:end_pos].strip()
        daily_plans[day_name] = day_content

    if not daily_plans:
        daily_plans = {"Full Plan": meal_plan_text}
    
    return daily_plans

def plot_nutrient_levels_for_day(day, day_plan_text):
    """
    Extract nutrient levels from the day's meal plan text, plot a bar chart,
    and if any key nutrient is missing, return a suggestions string.
    
    Expected nutrients: Calories, Protein, Carbs, Fat, Fiber.
    """
    expected_nutrients = ["Calories", "Protein", "Carbs", "Fat", "Fiber"]
    nutrient_values = {}
    
    for nutrient in expected_nutrients:
        pattern = rf"{nutrient}\s*:\s*([\d\.]+)"
        match = re.search(pattern, day_plan_text, re.IGNORECASE)
        if match:
            try:
                nutrient_values[nutrient] = float(match.group(1))
            except ValueError:
                pass

    missing_nutrients = [nutrient for nutrient in expected_nutrients if nutrient not in nutrient_values]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    if nutrient_values:
        ax.bar(nutrient_values.keys(), nutrient_values.values(), color='skyblue')
        ax.set_title(f"Nutrient Levels for {day}")
        ax.set_ylabel("Amount")
        ax.set_xlabel("Nutrient")
    else:
        ax.text(0.5, 0.5, "No nutrient data found", horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
    
    suggestion_text = ""
    if missing_nutrients:
        suggestions = {
            "Calories": "Consider adding an energy-rich snack like a granola bar.",
            "Protein": "Consider a protein snack such as Greek yogurt or a protein shake.",
            "Carbs": "Consider a carb-based snack like a piece of fruit or whole grain crackers.",
            "Fat": "Consider a healthy fat source like nuts or avocado toast.",
            "Fiber": "Consider fiber-rich options like vegetables or whole grains."
        }
        suggestion_text = "Missing Nutrients:\n"
        for nutrient in missing_nutrients:
            suggestion_text += f"- {nutrient}: {suggestions.get(nutrient, 'Consider a nutrient-rich snack.')}\n"
    
    return fig, suggestion_text

# ============================
# Functions for Food Recognition & Logging
# ============================
def get_base64_image(uploaded_file):
    """Encode the uploaded file to a base64 string."""
    bytes_data = uploaded_file.getvalue()
    base64_image = base64.b64encode(bytes_data).decode('utf-8')
    return base64_image

def get_nutritional_breakdown(base64_image):
    """
    Call Mistral API with the base64-encoded image to get a nutritional breakdown.
    The prompt asks for a JSON response including keys: Calories, Protein, Carbs, Fat, and Fiber.
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        st.error("MISTRAL_API_KEY not set in environment variables.")
        return None

    client = Mistral(api_key=api_key)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        '''Whats in this image? Please provide a full nutritional breakdown of this meal in JSON format. Output JSON without any additional text. This is what the JSON output should look like. The Numbers should be operable float numbers "meal": "Pancakes with Toppings",
  "ingredients": {
    "pancakes": {
      "calories": 300,
      "carbohydrates": 50,
      "protein": 5,
      "fat": 5"'''
                    )
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }
    ]
    
    try:
        chat_response = client.chat.complete(
            model=FOOD_MODEL,
            messages=messages
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        st.error(f"API call failed: {e}")
        return None

def log_food(nutritional_info):
    """
    Log the nutritional info of a food item into the daily food intake log stored in session_state.
    """
    if "food_log" not in st.session_state:
        st.session_state.food_log = []
    st.session_state.food_log.append(nutritional_info)

def plot_food_log():
    """
    Aggregate the logged food items and create a bar chart that sums the key nutrients.
    """
    if "food_log" not in st.session_state or not st.session_state.food_log:
        st.write("No food logged yet.")
        return
    
    keys = ["Calories", "Protein", "Carbs", "Fat", "Fiber"]
    aggregated = {key: 0 for key in keys}
    
    for entry in st.session_state.food_log:
        for key in keys:
            try:
                aggregated[key] += float(entry.get(key, 0))
            except (ValueError, TypeError):
                pass
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(aggregated.keys(), aggregated.values(), color='skyblue')
    ax.set_title("Daily Nutrient Totals")
    ax.set_ylabel("Total Amount")
    ax.set_xlabel("Nutrient")
    st.pyplot(fig)
    
    st.write("Aggregated Nutritional Info:", aggregated)

# ============================
# Main App: Sidebar Navigation
# ============================
def main():
    st.title("Multi-Feature Nutrition App")
    
    app_mode = st.sidebar.radio("Select Mode", ["Meal Plan Generator", "Food Recognition & Logging"])
    
    if app_mode == "Meal Plan Generator":
        st.header("Generate Your Meal Plan")
        # Initialize session state for meal plan
        if 'meal_plan' not in st.session_state:
            st.session_state.meal_plan = None
        if 'daily_plans' not in st.session_state:
            st.session_state.daily_plans = {}
        if 'current_day' not in st.session_state:
            st.session_state.current_day = None
        
        # User Profile Setup
        st.sidebar.subheader("User Profile Setup")
        age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30)
        gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
        weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        activity = st.sidebar.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Active", "Very Active"])
        dietary = st.sidebar.multiselect("Dietary Preferences", ["Vegan", "Vegetarian", "Keto", "Halal", "Gluten-Free", "None"])
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
        
        if st.button("Generate 7-Day Meal Plan"):
            with st.spinner("Generating meal plan..."):
                meal_plan = generate_meal_plan(profile)
                st.session_state.meal_plan = meal_plan
                daily_plans = parse_meal_plan_by_day(meal_plan)
                st.session_state.daily_plans = daily_plans
                if daily_plans:
                    st.session_state.current_day = list(daily_plans.keys())[0]
        
        if st.session_state.meal_plan:
            st.subheader("Navigate Your Meal Plan")
            cols = st.columns(len(st.session_state.daily_plans))
            for i, day in enumerate(st.session_state.daily_plans.keys()):
                if cols[i].button(day):
                    st.session_state.current_day = day
            if st.session_state.current_day:
                st.markdown(f"### {st.session_state.current_day}'s Meal Plan")
                st.markdown(st.session_state.daily_plans[st.session_state.current_day])
                fig, suggestion_text = plot_nutrient_levels_for_day(
                    st.session_state.current_day,
                    st.session_state.daily_plans[st.session_state.current_day]
                )
                st.pyplot(fig)
                if suggestion_text:
                    st.markdown("### Suggestions to Fill Nutrient Gaps")
                    st.write(suggestion_text)
            with st.expander("View Complete Meal Plan"):
                st.markdown("### Your Complete 7-Day Meal Plan")
                st.write(st.session_state.meal_plan)
    
    elif app_mode == "Food Recognition & Logging":
        st.header("Food Recognition & Nutritional Breakdown")
        uploaded_file = st.file_uploader("Upload an image of your meal", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Meal", use_column_width=True)
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    base64_image = get_base64_image(uploaded_file)
                    result = get_nutritional_breakdown(base64_image)
                if result:
                    st.subheader("Nutritional Breakdown (JSON)")
                    st.code(result, language="json")
                    # Try parsing the JSON (if it's valid)
                    try:
                        nutritional_info = json.loads(result)
                    except Exception as e:
                        st.error(f"Error parsing JSON: {e}")
                        nutritional_info = None
                    if nutritional_info:
                        if st.button("Log this Food"):
                            log_food(nutritional_info)
                            st.success("Food logged successfully!")
        st.header("Daily Food Intake Log")
        plot_food_log()

if __name__ == "__main__":
    main()
