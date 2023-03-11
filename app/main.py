from fastapi import FastAPI
import gradio as gr
import pandas as pd
import numpy as np
import locale

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

app = FastAPI()

# Load the primary dataset, stored as a publicly available Google Drive file
url='https://drive.google.com/file/d/1Aw9QEfKAOXWUVVZjZgKhPsv8U9HS-tcE/view?usp=sharing'
file_id=url.split('/')[-2]
dwn_url='https://drive.google.com/uc?id=' + file_id
df = pd.read_csv(dwn_url)

# Load a complimentary dataset used for looking up Latitude/Longitude from Postal Code
url='https://drive.google.com/file/d/1ZSmEg3saVsp6J6XaAXDGdYaCys6RUa-_/view?usp=sharing'
file_id=url.split('/')[-2]
dwn_url='https://drive.google.com/uc?id=' + file_id
postal_df = pd.read_csv(dwn_url)

# Use postal dataframe and merge it into the main dataframe using an sql-like left join
postal_df['zip'] = postal_df['POSTAL_CODE']
postal_df = postal_df[['zip', 'LATITUDE', 'LONGITUDE']]
df = df.merge(postal_df, how='left', on='zip')

# Drop missing values for clustering
df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])

fuel_mappings = {
    "Electric / Premium Unleaded" : "Electric",
    "E85" : "E85 / Unleaded",
    "E85 / Unleaded; Unleaded" : "E85 / Unleaded",
    "Unleaded / E85" : "E85 / Unleaded",
    "Unleaded / Unleaded" : "Unleaded",
    "E85 / Unleaded; Unleaded / Unleaded" : "Unleaded",
    "Compressed Natural Gas / Lpg" : "Compressed Natural Gas",
    "Compressed Natural Gas / Unleaded" : "Compressed Natural Gas",
    "Biodiesel" : "Diesel",
    "E85 / Premium Unleaded" : "E85 / Unleaded",
    "Electric / E85" : "Electric",
    "Compressed Natural Gas" : "Compressed Natural Gas",
    "Compressed Natural Gas; Unleaded" : "Compressed Natural Gas",
    "Unleaded / Premium Unleaded" : "Unleaded",
    "Unleaded / Electric" : "Electric",
    "Electric / Hydrogen" : "Electric",
    "Premium Unleaded / Natural Gas" : "Unleaded",
    "Diesel / Premium Unleaded" : "Diesel",
    "Diesel" : "Diesel",
    "Electric" : "Electric",
    "Unleaded" : "Unleaded",
    "E85 / Unleaded" : "E85 / Unleaded",
    "Premium Unleaded" : "Unleaded",
    "Electric / Unleaded" : "Electric",
    "Premium Unleaded; Unleaded" : "Unleaded",
    "Premium Unleaded / Unleaded" : "Unleaded",
}

imputing_transformers = [
    ('miles', SimpleImputer(), ['miles']),
    ('year', SimpleImputer(), ['year']),
    ('make', 'passthrough', ['make']),
    ('model', 'passthrough', ['model']),
    ('trim', 'passthrough', ['trim']),
    ('body_type', 'passthrough', ['body_type']),
    ('vehicle_type', SimpleImputer(strategy='most_frequent'), ['vehicle_type']),
    ('drivetrain', SimpleImputer(strategy='most_frequent'), ['drivetrain']),
    ('transmission', SimpleImputer(strategy='most_frequent'), ['transmission']),
    ('fuel_type', FunctionTransformer(lambda col: col.replace(fuel_mappings)), ['fuel_type']),
    ('engine_size', SimpleImputer(), ['engine_size']),
    ('engine_block', SimpleImputer(strategy='most_frequent'), ['engine_block']),
    ('latitude', SimpleImputer(), ['LATITUDE']),
    ('longitude', SimpleImputer(), ['LONGITUDE']),
]

encoding_transformers = [
    ('miles', 'passthrough', [0]),
    ('year', 'passthrough', [1]),
    ('make', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1), [2]),
    ('model', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1), [3]),
    ('trim', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1), [4]),
    ('body_type', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1), [5]),
    ('vehicle_type', OneHotEncoder(drop='first'), [6]),
    ('drivetrain', OneHotEncoder(), [7]),
    ('transmission', OneHotEncoder(drop='first'), [8]),
    ('fuel_type', OneHotEncoder(), [9]),
    ('engine_size', 'passthrough', [10]),
    ('engine_block', OneHotEncoder(), [11]),
    ('latitude', 'passthrough', [12]),
    ('longitude', 'passthrough', [13]),
]

class Preprocessor:
  def __init__(self):
    self.imputer = ColumnTransformer(imputing_transformers)
    self.encoder = ColumnTransformer(encoding_transformers)
    self.kMeans = KMeans(n_clusters=6)
    self.stdScaler = StandardScaler()

  def fit_transform(self, X):
    X = self.imputer.fit_transform(X)
    X = self.encoder.fit_transform(X)
    self.kMeans.fit(X[:, -2:])
    X = np.hstack(( X, np.expand_dims(self.kMeans.predict(X[:, -2:]), axis=1) ))
    X = self.stdScaler.fit_transform(X)

    return X

  def transform(self, X):
    X = self.imputer.transform(X)
    X = self.encoder.transform(X)
    X = np.hstack(( X, np.expand_dims(self.kMeans.predict(X[:, -2:]), axis=1) ))
    X = self.stdScaler.transform(X)

    return X

df = df[df['price'] > 0]

X_train, X_val, y_train, y_val = train_test_split(df, df['price'], test_size=0.2, random_state=1)
preprocessor = Preprocessor()

X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)

"""# Train some Models"""
from sklearn.ensemble import RandomForestRegressor
rand_for = RandomForestRegressor(random_state=42).fit(X_train, y_train)
rand_for.score(X_val, y_val)

# Reload the dataframe, we will need it later
url='https://drive.google.com/file/d/1Aw9QEfKAOXWUVVZjZgKhPsv8U9HS-tcE/view?usp=sharing'
file_id=url.split('/')[-2]
dwn_url='https://drive.google.com/uc?id=' + file_id
web_df = pd.read_csv(dwn_url)

# Set locale for currency formatting
locale.setlocale(locale.LC_ALL, '')

# Function that will be run to make the ML prediction
def make_prediction(*inputs):
  column_labels = ['miles', 'year', 'make', 'model', 'trim', 'body_type', 'vehicle_type', 'drivetrain', 'transmission', 'fuel_type', 'engine_size', 'engine_block', 'zip']
  pred_df = pd.DataFrame(data=[inputs], columns=column_labels)
  pred_df = pred_df.replace('', np.nan)
  pred_df['zip'] = pred_df['zip'].astype('str').str.upper().str.replace(' ','').apply(lambda zip: zip[:2] + ' ' + zip[2:])
  pred_df = pred_df.merge(postal_df, how='left', on='zip')

  input = preprocessor.transform(pred_df)
  price = rand_for.predict(input)[0]

  return f"## Your Predicted price is {locale.currency(price, grouping=True)}"

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
  gr.Markdown('''
  # Used Car Price Prediction
  Welcome to our machine learning project where we try to predict the value of your car given some input parameters.

  We are using a Canadian used car listing [dataset from Kaggle](https://www.kaggle.com/datasets/rupeshraundal/marketcheck-automotive-data-us-canada?select=ca-dealers-used.csv). Check out our [Colab Notebook](https://colab.research.google.com/drive/1a9Sn1ooKsqXIAhT_MadAuZ-1xJ1dz-p7?usp=sharing) and our [GitHub Repo](https://github.com/petermaksymo/car-price-prediction).
  ''')

  with gr.Row():
    # Mileage
    miles = gr.Number(label="Mileage")

    # Year
    year = gr.Number(label="Year")

    # Make, list all choices
    make_choices = np.sort(web_df['make'].dropna().unique()).tolist()
    make = gr.Dropdown(label="Make", choices=make_choices)

    # Model: render when make is selected, only show models which correspond to the make
    model = gr.Dropdown(label="Model", choices=[], interactive=False)
    make.change(lambda x: filter_choices(x, 'make', 'model'), make, model)

  with gr.Row():
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

    # Postal Code: Will add latitude and longitude in preprocessing
    postal_code = gr.Textbox(label='Postal Code', placeholder='ex. A1A 1A1')

  inputs = [miles, year, make, model, trim, body_type, vehicle_type, drivetrain, transmission, fuel_type, engine_size, engine_block, postal_code]
  output = gr.Markdown()
  btn = gr.Button("Predict Price")
  btn.click(fn=make_prediction, inputs=inputs, outputs=output)

  gr.Markdown('''
  <br/><br/>
  This project was created by [Peter Maksymowsky](https://www.linkedin.com/in/peter-maksymowsky/), [Oleksandra Nahorna](https://www.linkedin.com/in/oleksandra-nahorna/), and [Henry Chukwunonso Nwokoye](https://www.linkedin.com/in/henry-chukwunonso-phd/).
  ''')


app = gr.mount_gradio_app(app, demo, path='/', show_api=False)
