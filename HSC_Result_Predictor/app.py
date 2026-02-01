#gradio app 

import gradio as gr
import pandas as pd
import pickle
import numpy as np

# 1. Load the Model
with open("student_rf_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# 2. The Logic Function
def predict_gpa(gender, age, address, famsize, 
                Pstatus, M_Edu, F_Edu, M_Job, F_Job, 
                relationship, smoker, tuition_fee, time_friends,
                  ssc_result):
    
    # Pack inputs into a DataFrame
    # The column names must match your CSV file exactly
    input_df = pd.DataFrame([[
        gender, age, address, famsize, Pstatus, 
        M_Edu, F_Edu, M_Job, F_Job, relationship, 
        smoker, tuition_fee, time_friends, ssc_result

    ]],
      columns=[
        'gender', 'age', 'address', 'famsize', 'Pstatus', 'M_Edu', 'F_Edu', 'M_Job', 'F_Job', 'relationship', 'smoker', 'tuition_fee', 'time_friends', 'ssc_result'
    ])
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    # Return formatted result (Clipped 0-5)
    return f"Predicted HSC Result: {np.clip(prediction, 0, 5):.2f}"

# 3. The App Interface
# Defining inputs in a list to keep it clean
inputs = [
    gr.Radio(["M", "F"], label="Gender"),
    gr.Number(label="Age", value=18),
    gr.Radio(["Urban", "Rural"], label="Address"),
    gr.Radio(["GT3", "LE3"], label="Family Size"),
    gr.Radio(["Together", "Apart"], label="Parent Status"),
    gr.Slider(0, 4, step=1, label="Mother's Edu"),
    gr.Slider(0, 4, step=1, label="Father's Edu"),
    gr.Dropdown(["At_home", "Health", "Other", "Services", "Teacher"], label="Mother's Job"),
    gr.Dropdown(["Teacher", "Other", "Services", "Health", "Business", "Farmer"], label="Father's Job"),
    gr.Radio(["Yes", "No"], label="Relationship"),
    gr.Radio(["Yes", "No"], label="Smoker"),
    gr.Number(label="Tuition Fee"),
    gr.Slider(1, 5, step=1, label="Time with Friends"),
    gr.Number(label="SSC Result (GPA)")
]

app = gr.Interface(
    fn=predict_gpa,
      inputs=inputs,
        outputs="text", 
        title="HSC Predictor")

app.launch(share=True)

