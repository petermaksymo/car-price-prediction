from fastapi import FastAPI
import gradio as gr
import pandas as pd
import numpy as np

app = FastAPI()

# Reload the dataframe, we will need it later
url='https://drive.google.com/file/d/1Aw9QEfKAOXWUVVZjZgKhPsv8U9HS-tcE/view?usp=sharing'
file_id=url.split('/')[-2]
dwn_url='https://drive.google.com/uc?id=' + file_id
raw_df = pd.read_csv(dwn_url)

def process_df_for_web(raw_df):
  df = raw_df.drop(columns=['id', 'vin', 'price', 'stock_no', 'seller_name'])

  return df
web_df = process_df_for_web(raw_df)

# Function that will be run to make the ML prediction
# TODO: Add preprocessing for sample and complete function
def make_prediction(*inputs):
  print(inputs)
  return "$1000"

# Helper function that let's us reduce the choices for a Dropdown given a prior choice
def filter_choices(choice, choice_col, filter_col):
    # filter df into only rows where the choice has been made in choice_col
    choices = web_df[web_df[choice_col] == choice][filter_col].dropna().unique().astype('str')

    # if only one value, pre-fill the value
    if(len(choices) == 1):
      return gr.Dropdown.update(choices=choices.tolist(), value=choices[0], visible=True, interactive=False)
    
    # sort choices alphabetically
    choices = np.sort(choices).tolist()
    return gr.Dropdown.update(choices=choices, visible=True, interactive=True)


with gr.Blocks() as demo:
  with gr.Row():
    with gr.Column(scale=3):
      # Mileage
      miles = gr.Textbox(label="Mileage")

      # Year
      year = gr.Number(label="Year")

      # Make, list all choices
      make_choices = np.sort(web_df['make'].dropna().unique()).tolist()
      make = gr.Dropdown(label="Make", choices=make_choices)

      # Model: render when make is selected, only show models which correspond to the make
      model = gr.Dropdown(label="Model", choices=[], visible=False)
      make.change(lambda x: filter_choices(x, 'make', 'model'), make, model)

      # Trim: render when model is selected, only show trims which correspond to the model
      trim = gr.Dropdown(label="Trim", choices=[], visible=False)
      model.change(lambda x: filter_choices(x, 'model', 'trim'), model, trim)

      # Body Type: render when model is selected, only show body types which correspond to the model
      body_type = gr.Dropdown(label="Body Type", choices=[], visible=False)
      model.change(lambda x: filter_choices(x, 'model', 'body_type'), model, body_type)

      # Vehicle Type: render when model is selected, only show vehicle types which correspond to the model
      vehicle_type = gr.Dropdown(label="Vehicle Type", choices=[], visible=False)
      model.change(lambda x: filter_choices(x, 'model', 'vehicle_type'), model, vehicle_type)

      # Drivetrain: render when model is selected, only show drivetrains which correspond to the model
      drivetrain = gr.Dropdown(label="Drivetrain", choices=[], visible=False)
      model.change(lambda x: filter_choices(x, 'model', 'drivetrain'), model, drivetrain)

      # Transmission: render when model is selected, only show transmissions which correspond to the model
      transmission = gr.Dropdown(label="Transmission", choices=[], visible=False)
      model.change(lambda x: filter_choices(x, 'model', 'transmission'), model, transmission)

      # Fuel Type: render when model is selected, only show fuel types which correspond to the model
      fuel_type = gr.Dropdown(label="Fuel Type", choices=[], visible=False)
      model.change(lambda x: filter_choices(x, 'model', 'fuel_type'), model, fuel_type)

      # Engine Size: render when model is selected, only show engine sizes which correspond to the model
      engine_size = gr.Dropdown(label="Engine Size", choices=[], visible=False)
      model.change(lambda x: filter_choices(x, 'model', 'engine_size'), model, engine_size)

      # Engine Block: render when model is selected, only show engine blocks which correspond to the model
      engine_block = gr.Dropdown(label="Engine Block", choices=[], visible=False)
      model.change(lambda x: filter_choices(x, 'model', 'engine_block'), model, engine_block)

      # TODO: Add location question once we figure out how we will preprocess it

    with gr.Column(scale=1):
      inputs = [miles, year, make, model, trim, body_type, vehicle_type, drivetrain, transmission, fuel_type, engine_size, engine_block]
      output = gr.Textbox()
      
  with gr.Row():
    btn = gr.Button("Predict Price")
    btn.click(fn=make_prediction, inputs=inputs, outputs=output)


app = gr.mount_gradio_app(app, demo, path='/')
