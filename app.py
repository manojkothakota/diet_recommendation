import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import random, os, requests

st.set_page_config("AI Diet Planner","🥗")
st.title("🥗 Autonomous AI Clinical Diet Planner")

df=pd.read_csv("clean_recipes_10k.csv")
print(df.head())
print(df.shape)
NUTRIENTS = ['Calories','FatContent','SaturatedFatContent','CholesterolContent',
             'SodiumContent','CarbohydrateContent','FiberContent','SugarContent','ProteinContent']

# ---------------- Motivation Brain ----------------
MOTIVATION = {
    "Diabetes": ["Low sugar eating is self-care.","Every meal heals your body."],
    "BP": ["Low salt = strong heart.","Choose food that calms your heart."],
    "Heart": ["Food is your daily medicine.","Your heart loves your choices."],
    "Obesity": ["You are redesigning your life.","Small steps make big change."],
    "Fitness": ["Strong body starts with smart food.","Champions eat smart."]
}

# ---------------- Groq AI Motivation ----------------
def groq_motivation(disease):
    try:
        headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                   "Content-Type": "application/json"}
        prompt = f"Give one motivational health message for a {disease} patient."
        data = {"model":"llama3-8b-8192","messages":[{"role":"user","content":prompt}]}
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",json=data,headers=headers)
        return r.json()['choices'][0]['message']['content']
    except:
        return random.choice(MOTIVATION[disease])

# ---------------- Calorie AI ----------------
def calculate_calories(age, gender, weight, height, goal):
    if gender=="Male":
        bmr = 88.36 + 13.4*weight + 4.8*height - 5.7*age
    else:
        bmr = 447.6 + 9.2*weight + 3.1*height - 4.3*age
    if goal=="Lose Weight": bmr*=0.8
    elif goal=="Gain Weight": bmr*=1.2
    return int(bmr)

# ---------------- Disease Rules ----------------
DISEASE_RULES = {
    "Diabetes":[25,25,1500,200],
    "BP":[30,30,1200,200],
    "Heart":[25,25,1200,150],
    "Obesity":[20,20,1000,150],
    "Fitness":[40,40,2000,300]
}

# ---------------- ML Recommender ----------------
def recommend(df,target,disease):
    fat,sugar,sodium,chol = DISEASE_RULES[disease]
    data = df[(df.FatContent<=fat)&(df.SugarContent<=sugar)&
              (df.SodiumContent<=sodium)&(df.CholesterolContent<=chol)]
    scaler = StandardScaler()
    X = scaler.fit_transform(data[NUTRIENTS])
    model = NearestNeighbors(metric="cosine")
    model.fit(X)
    idx = model.kneighbors(scaler.transform([target]),4,return_distance=False)[0]
    return data.iloc[idx]

# ---------------- UI ----------------
age = st.number_input("Age",10,100,25)
gender = st.selectbox("Gender",["Male","Female"])
height = st.number_input("Height (cm)",120,220,170)
weight = st.number_input("Weight (kg)",30,200,70)
goal = st.selectbox("Goal",["Lose Weight","Maintain","Gain Weight"])
disease = st.selectbox("Health Condition",list(DISEASE_RULES.keys()))

if st.button("Generate My AI Diet Plan"):
    calories = calculate_calories(age,gender,weight,height,goal)
    st.success(f"🔥 Your AI-calculated daily calories: {calories} kcal")
    st.info("💡 "+groq_motivation(disease))

    target = [calories,30,10,150,1500,250,30,25,100]
    recs = recommend(df,target,disease)
    meals = ["Breakfast","Lunch","Snack","Dinner"]

    for meal,(_,r) in zip(meals,recs.iterrows()):
        st.subheader(f"{meal}: {r['Name']}")
        st.write(f"Calories: {int(r['Calories'])} kcal")
        st.write(f"Protein: {r['ProteinContent']}g | Fat: {r['FatContent']}g | Sugar: {r['SugarContent']}g")
        st.caption(r['RecipeInstructions'])

