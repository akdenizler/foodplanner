import streamlit as st
import requests
import json
import re
import dotenv
import os
from PIL import Image
import matplotlib.pyplot as plt

dotenv.load_dotenv()

# === Configuration for Mistral API ===
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
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
    daily_plans = {}
    # Regular expression to find day headers (DAY X: DAY_NAME)
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

def recognize_food(image_bytes):
    """
    Placeholder for food recognition. Mistral API doesn't support image recognition for food.
    """
    return "Food recognition functionality using Mistral API is not available. Please use an alternative service."

def plot_nutrient_levels_for_day(day, day_plan_text):
    """
    Extract nutrient levels from the day's meal plan text, plot a bar chart,
    and if any key nutrient is missing, return a suggestions string.
    
    Expected nutrients: Calories, Protein, Carbs, Fat, Fiber.
    """
    expected_nutrients = ["Calories", "Protein", "Carbs", "Fat", "Fiber"]
    nutrient_values = {}

    # Look for patterns like "Protein: 30" in the text (units are ignored)
    for nutrient in expected_nutrients:
        pattern = rf"{nutrient}\s*:\s*([\d\.]+)"
        match = re.search(pattern, day_plan_text, re.IGNORECASE)
        if match:
            try:
                nutrient_values[nutrient] = float(match.group(1))
            except ValueError:
                pass

    missing_nutrients = [nutrient for nutrient in expected_nutrients if nutrient not in nutrient_values]
    
    # Create bar chart
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

# === Streamlit UI ===
def main():
    st.title("🍽 AI-Powered Meal Plan Generator")
    
    # Initialize session state for storing the meal plan
    if 'meal_plan' not in st.session_state:
        st.session_state.meal_plan = None
    if 'daily_plans' not in st.session_state:
        st.session_state.daily_plans = {}
    if 'current_day' not in st.session_state:
        st.session_state.current_day = None
    
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
        st.subheader("Navigate Your Meal Plan")
        cols = st.columns(len(st.session_state.daily_plans))
        for i, day in enumerate(st.session_state.daily_plans.keys()):
            if cols[i].button(day):
                st.session_state.current_day = day
        
        if st.session_state.current_day:
            st.markdown(f"### {st.session_state.current_day}'s Meal Plan")
            st.markdown(st.session_state.daily_plans[st.session_state.current_day])
            
            # Plot nutrient levels for the selected day and show suggestions if needed
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
    
    # Food Recognition Section
    st.header("Food Recognition & Logging")
    uploaded_file = st.file_uploader("Upload an image of your meal", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Meal", use_column_width=True)
        
        if st.button("Recognize Food"):
            with st.spinner("Analyzing your food..."):
                food_info = recognize_food(uploaded_file.getvalue())
                st.markdown("### Food Information")
                st.write(food_info)
    
    # Meal Plan History
    st.header("Meal Plan History")
    st.write("This feature is not implemented yet. In a complete solution, you'd save and display past meal plans here.")

if __name__ == "__main__":
    main()
