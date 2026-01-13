import streamlit as st
import pandas as pd
import numpy as np
import random, os, requests

st.set_page_config("AI Diet Planner","🥗")
st.title("🥗 Autonomous AI Clinical Diet Planner")

df = pd.read_csv("clean_recipes_10k.csv")

# ---------------- Nutrient Columns ----------------
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

# ---------------- Groq Motivation ----------------
def groq_motivation(disease):
    try:
        headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}","Content-Type": "application/json"}
        data = {"model":"llama3-8b-8192","messages":[{"role":"user","content":f"Give one motivational message for {disease} patient"}]}
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",json=data,headers=headers)
        return r.json()['choices'][0]['message']['content']
    except:
        return random.choice(MOTIVATION[disease])

# ---------------- Calorie Brain ----------------
def calculate_calories(age, gender, weight, height, goal):
    bmr = 88.36 + 13.4*weight + 4.8*height - 5.7*age if gender=="Male" else 447.6 + 9.2*weight + 3.1*height - 4.3*age
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

# ---------------- AI Engine ----------------
def normalize(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)

def cosine(a,b):
    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))

def recommend(df,target,disease):
    fat,sugar,sodium,chol = DISEASE_RULES[disease]
    data = df[(df.FatContent<=fat)&(df.SugarContent<=sugar)&
              (df.SodiumContent<=sodium)&(df.CholesterolContent<=chol)]
    if len(data) < 4: data = df.copy()
    X = normalize(data[NUTRIENTS].values.astype(float))
    t = normalize(np.array(target).reshape(1,-1))[0]
    scores = [cosine(t,row) for row in X]
    best = np.argsort(scores)[-4:]
    return data.iloc[best]

# ---------------- UI ----------------
age = st.number_input("Age",10,100,25)
gender = st.selectbox("Gender",["Male","Female"])
height = st.number_input("Height (cm)",120,220,170)
weight = st.number_input("Weight (kg)",30,200,70)
goal = st.selectbox("Goal",["Lose Weight","Maintain","Gain Weight"])
disease = st.selectbox("Health Condition",list(DISEASE_RULES.keys()))

if st.button("Generate My AI Diet Plan"):
    calories = calculate_calories(age,gender,weight,height,goal)
    st.success(f"🔥 Your daily AI calories: {calories} kcal")
    st.info("💡 "+groq_motivation(disease))

    target = [calories,30,10,150,1500,250,30,25,100]
    recs = recommend(df,target,disease)
    meals = ["Breakfast","Lunch","Snack","Dinner"]

    for meal,(_,r) in zip(meals,recs.iterrows()):
        st.subheader(f"{meal}: {r['Name']}")
        st.write(f"Calories: {int(r['Calories'])} kcal")
        st.write(f"Protein: {r['ProteinContent']}g | Fat: {r['FatContent']}g | Sugar: {r['SugarContent']}g")
        st.caption(r['RecipeInstructions'])
