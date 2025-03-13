import streamlit as st
import requests
import re
from PIL import Image
import io
import datetime
import pandas as pd
import uuid
import os
import base64
import dotenv
from mistralai import Mistral
import json

dotenv.load_dotenv()
# === Configuration for APIs ===
# Store API keys in environment variables or Streamlit secrets
api_key = os.environ.get("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

def get_mistral_api_key():
    """Helper function to get the Mistral API key."""
    return os.environ.get("MISTRAL_API_KEY")

# === Collapsible Meal Plan Display (New) ===
def display_collapsible_meal_plan(day_content):
    """
    Minimal example that places each meal in an expander.
    Replace these placeholders with real parsing logic if desired.
    """
    with st.expander("Breakfast"):
        st.write("Placeholder for breakfast details.")
    with st.expander("Lunch"):
        st.write("Placeholder for lunch details.")
    with st.expander("Dinner"):
        st.write("Placeholder for dinner details.")
    with st.expander("Snacks"):
        st.write("Placeholder for snacks details.")
    
    # Finally, show the raw AI-generated text if you want
    with st.expander("Raw Plan Text"):
        st.markdown(day_content)

# === API Call Functions ===
def generate_meal_plan(profile):
    """Generate a personalized meal plan using Mistral AI API with additional user preferences."""
    api_key = get_mistral_api_key()
    if not api_key:
        return "Error: Mistral API key not found. Please set up your API key."
    
    # Build prompt based on profile and additional preferences
    prompt = (
        f"Generate a personalized 7-day meal plan for a {profile['age']} year old {profile['gender']} "
        f"with weight {profile['weight']}kg, height {profile['height']}cm, activity level {profile['activity']}, "
        f"dietary preferences {', '.join(profile['dietary'])}, "
        f"Add this series of special symbols at the end of each day's meal plan '-=*=-"
    )
    
    # Add menstrual cycle info for females (if applicable)
    if profile['gender'] == "Female" and profile['menstrual_cycle'] != "Not Applicable":
       prompt += (
    f"User is in menstrual cycle phase {profile['menstrual_cycle']} "
    "give suggestions to support hormonal health."
)

    
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
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        meal_plan = result["choices"][0]["message"]["content"]
        return meal_plan
    except requests.exceptions.RequestException as e:
        return f"API Request Error: {str(e)}"
    except KeyError as e:
        return f"API Response Format Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

def parse_meal_plan_by_day(meal_plan_text):
    """
    Parse the meal plan text to extract daily meal plans using the marker "-=*=-" as an optional end delimiter.
    Returns a dictionary with day names as keys and the corresponding meal content as values.
    """
    daily_plans = {}
    # This pattern will capture each day's section until it finds the marker "-=*=-" or the end of the text.
    pattern = r"(?i)(DAY\s+\d+[\s:.-]*([A-Za-z]+))\s*(.*?)(?:\s*-=\*=-|$)"
    
    matches = re.finditer(pattern, meal_plan_text, re.DOTALL)
    for match in matches:
        header = match.group(1).strip()
        day_name = match.group(2).capitalize()
        day_content = match.group(3).strip()
        daily_plans[day_name] = f"{header}\n{day_content}"
    
    if not daily_plans:
        daily_plans = {"Full Plan": meal_plan_text}
    
    return daily_plans


def encode_image_from_bytes(image_bytes):
    """Encode the image bytes to base64."""
    try:
        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def recognize_food(image_bytes):
    """Recognize food in an image using Mistral's multimodal API."""
    try:
        base64_image = encode_image_from_bytes(image_bytes)
        if not base64_image:
            return "Error: Failed to encode image."
        
        client = Mistral(api_key=api_key)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """What's in this image? Please provide a full nutritional breakdown of this meal in JSON format. Output JSON without any additional text. This is what the JSON output should look like. The Numbers should be operable float numbers 
                        {
                            "meal": "Meal name",
                            "ingredients": {
                                "ingredient1": {
                                    "calories": 300,
                                    "carbohydrates": 50,
                                    "protein": 5,
                                    "fat": 5
                                },
                                "ingredient2": {
                                    "calories": 150,
                                    "carbohydrates": 20,
                                    "protein": 10,
                                    "fat": 5
                                }
                            },
                            "total": {
                                "calories": 450,
                                "carbohydrates": 70,
                                "protein": 15, 
                                "fat": 10
                            }
                        }"""
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ]
        
        chat_response = client.chat.complete(
            model="pixtral-12b-2409",
            messages=messages
        )
        response_content = chat_response.choices[0].message.content
        
        try:
            json_content = response_content
            if not response_content.strip().startswith('{'):
                json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1)
                else:
                    json_match = re.search(r'(\{.*\})', response_content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1)
            
            nutritional_data = json.loads(json_content)
            detected_items = list(nutritional_data.get('ingredients', {}).keys())
            if not detected_items and 'meal' in nutritional_data:
                detected_items = [nutritional_data['meal']]
            
            nutritional_analysis = format_nutritional_data(nutritional_data)
            
            return {
                "detected_items": detected_items,
                "nutritional_analysis": nutritional_analysis,
                "raw_data": nutritional_data
            }
        except json.JSONDecodeError:
            return {
                "detected_items": ["Unknown"],
                "nutritional_analysis": "Could not parse nutritional data. Here's the raw response:\n\n" + response_content
            }
            
    except Exception as e:
        return f"Error recognizing food: {str(e)}"

def format_nutritional_data(data):
    """Format the nutritional data from JSON to a readable format."""
    result = []
    if 'meal' in data:
        result.append(f"## {data['meal']}")
    if 'ingredients' in data:
        result.append("### Ingredients:")
        for ingredient, nutrients in data['ingredients'].items():
            result.append(f"#### {ingredient.title()}")
            for nutrient, value in nutrients.items():
                unit = "g" if nutrient != "calories" else "kcal"
                result.append(f"- {nutrient.title()}: {value} {unit}")
    if 'total' in data:
        result.append("### Total Nutritional Value:")
        for nutrient, value in data['total'].items():
            unit = "g" if nutrient != "calories" else "kcal"
            result.append(f"- {nutrient.title()}: {value} {unit}")
    return "\n".join(result)

def save_meal_plan(profile, meal_plan):
    """Save the generated meal plan to history."""
    if 'meal_plan_history' not in st.session_state:
        st.session_state.meal_plan_history = []
    
    plan_id = str(uuid.uuid4())[:8]
    profile_summary = f"{profile['gender']}, {profile['age']}yo, {profile['weight']}kg, {profile['height']}cm, {profile['fitness_goal']}"
    meal_plan_entry = {
        "id": plan_id,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "profile": profile_summary,
        "details": profile,
        "plan": meal_plan,
        "daily_plans": parse_meal_plan_by_day(meal_plan)
    }
    
    st.session_state.meal_plan_history.insert(0, meal_plan_entry)
    return plan_id

def load_meal_plan_from_history(plan_id):
    """Load a specific meal plan from history."""
    if 'meal_plan_history' not in st.session_state:
        return None
    for plan in st.session_state.meal_plan_history:
        if plan["id"] == plan_id:
            return plan
    return None

# === Streamlit UI ===
def main():
    st.title("🍽 AI-Powered Meal Plan Generator")
    
    if 'meal_plan' not in st.session_state:
        st.session_state.meal_plan = None
    if 'daily_plans' not in st.session_state:
        st.session_state.daily_plans = {}
    if 'current_day' not in st.session_state:
        st.session_state.current_day = None
    if 'meal_plan_history' not in st.session_state:
        st.session_state.meal_plan_history = []
    if 'current_plan_id' not in st.session_state:
        st.session_state.current_plan_id = None
    if 'history_current_day' not in st.session_state:
        st.session_state.history_current_day = None
    
    tab1, tab2 = st.tabs(["Meal Plan Generator", "Meal Plan History"])
    
    with tab1:
        with st.sidebar:
            st.header("User Profile Setup")
            age = st.number_input("Age", min_value=10, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Female", "Male"])
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            additional_preferences = st.text_area("Additional Preferences", "Enter any foods you like/dislike, specific dietary needs, or other preferences here...").strip()
            activity = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Active", "Very Active"])
            dietary = st.multiselect("Dietary Preferences", ["Vegan", "Vegetarian", "Halal", "Kosher", "Gluten-Free", "None"])
            menstrual_cycle = "Not Applicable"
            if gender == "Female":
                menstrual_cycle = st.selectbox("Menstrual Cycle Phase", ["Not Applicable", "Follicular", "Ovulatory", "Luteal", "Menstrual"])
            fitness_goal = st.selectbox("Fitness Goals", ["Weight Loss", "Muscle Gain", "Maintenance"])
             
        
        profile = {
            "age": age,
            "gender": gender,
            "weight": weight,
            "height": height,
            "activity": activity,
            "dietary": dietary if dietary else ["None"],
            "menstrual_cycle": menstrual_cycle,
            "fitness_goal": fitness_goal,
            "additional_preferences": additional_preferences

        }
        
        st.header("Generate Your Meal Plan")
        col1, col2 = st.columns([3, 1])
        with col1:
            generate_button = st.button("Generate 7-Day Meal Plan", use_container_width=True)
        
        if generate_button:
            if not get_mistral_api_key():
                st.error("Missing Mistral API key. Please configure it to generate meal plans.")
            else:
                with st.spinner("Generating meal plan..."):
                    meal_plan = generate_meal_plan(profile)
                    if isinstance(meal_plan, str) and meal_plan.startswith("Error"):
                        st.error(meal_plan)
                    else:
                        st.session_state.meal_plan = meal_plan
                        daily_plans = parse_meal_plan_by_day(meal_plan)
                        st.session_state.daily_plans = daily_plans
                        if daily_plans:
                            st.session_state.current_day = list(daily_plans.keys())[0]
                        plan_id = save_meal_plan(profile, meal_plan)
                        st.session_state.current_plan_id = plan_id
                        st.success(f"Meal plan generated and saved to history (ID: {plan_id})!")
        
        # --- Display the selected day in collapsible sections ---
        if st.session_state.meal_plan and st.session_state.current_day:
            st.markdown(f"### {st.session_state.current_day}'s Meal Plan")
            display_collapsible_meal_plan(st.session_state.daily_plans[st.session_state.current_day])
            
            with st.expander("View Complete Meal Plan"):
                st.markdown("### Your Complete 7-Day Meal Plan")
                st.write(st.session_state.meal_plan)
        
        st.header("Food Recognition & Logging")
        uploaded_file = st.file_uploader("Upload an image of your meal", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Meal", use_column_width=True)
            if st.button("Analyze Food"):
                if not get_mistral_api_key():
                    st.error("Mistral API key is required for food analysis.")
                else:
                    with st.spinner("Analyzing your food..."):
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format=image.format)
                        img_byte_arr = img_byte_arr.getvalue()
                        food_info = recognize_food(img_byte_arr)
                        st.markdown("### Food Analysis Results")
                        if isinstance(food_info, dict):
                            if "detected_items" in food_info:
                                st.subheader("Detected Items")
                                st.write(", ".join(food_info["detected_items"]))
                            if "nutritional_analysis" in food_info:
                                st.subheader("Nutritional Analysis")
                                st.markdown(food_info["nutritional_analysis"])
                            if "raw_data" in food_info:
                                with st.expander("View Raw Data"):
                                    st.json(food_info["raw_data"])
                        else:
                            st.error(food_info)
    
    with tab2:
        st.header("Your Meal Plan History")
        if not st.session_state.meal_plan_history:
            st.info("No meal plans have been generated yet. Generate a meal plan to see it here.")
        else:
            history_data = []
            for plan in st.session_state.meal_plan_history:
                history_data.append({
                    "ID": plan["id"],
                    "Date": plan["date"],
                    "Profile": plan["profile"],
                })
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
            selected_plan_id = st.selectbox(
                "Select a meal plan to view:", 
                options=[plan["id"] for plan in st.session_state.meal_plan_history],
                format_func=lambda x: f"{x} - " + next((p["date"] + " (" + p["profile"] + ")" for p in st.session_state.meal_plan_history if p["id"] == x), "")
            )
            if selected_plan_id:
                selected_plan = load_meal_plan_from_history(selected_plan_id)
                if selected_plan:
                    st.subheader(f"Meal Plan {selected_plan['id']} - {selected_plan['date']}")
                    with st.expander("View Profile Details"):
                        profile_details = selected_plan["details"]
                        cols = st.columns(3)
                        cols[0].write(f"**Age:** {profile_details['age']}")
                        cols[0].write(f"**Gender:** {profile_details['gender']}")
                        cols[1].write(f"**Weight:** {profile_details['weight']} kg")
                        cols[1].write(f"**Height:** {profile_details['height']} cm")
                        cols[2].write(f"**Activity:** {profile_details['activity']}")
                        cols[2].write(f"**Goal:** {profile_details['fitness_goal']}")
                        st.write(f"**Dietary Preferences:** {', '.join(profile_details['dietary'])}")
                        if profile_details['gender'] == 'Female' and profile_details['menstrual_cycle'] != 'Not Applicable':
                            st.write(f"**Menstrual Cycle Phase:** {profile_details['menstrual_cycle']}")
                    
                    st.subheader("Navigate Days")
                    days = list(selected_plan["daily_plans"].keys())
                    num_days = len(days)
                    cols_per_row = min(7, num_days)
                    for i in range(0, num_days, cols_per_row):
                        row_days = days[i:i+cols_per_row]
                        cols = st.columns(len(row_days))
                        for j, day in enumerate(row_days):
                            if cols[j].button(day, key=f"history_day_{day}_{selected_plan_id}"):
                                st.session_state.history_current_day = day
                    
                    if st.session_state.history_current_day is None or st.session_state.history_current_day not in selected_plan["daily_plans"]:
                        st.session_state.history_current_day = days[0]
                    
                    current_day = st.session_state.history_current_day
                    st.markdown(f"### {current_day}'s Meal Plan")
                    # Use the new collapsible display here as well:
                    display_collapsible_meal_plan(selected_plan["daily_plans"][current_day])
                    
                    with st.expander("View Complete Meal Plan"):
                        st.markdown("### Complete 7-Day Meal Plan")
                        st.write(selected_plan["plan"])
                    
                    if st.button("Use This Plan Again", key=f"use_plan_{selected_plan_id}"):
                        st.session_state.meal_plan = selected_plan["plan"]
                        st.session_state.daily_plans = selected_plan["daily_plans"]
                        st.session_state.current_day = list(selected_plan["daily_plans"].keys())[0]
                        st.session_state.current_plan_id = selected_plan["id"]
                        st.success("Plan loaded into the current session! Switch to the Meal Plan Generator tab to view it.")

if __name__ == "__main__":
    main()
