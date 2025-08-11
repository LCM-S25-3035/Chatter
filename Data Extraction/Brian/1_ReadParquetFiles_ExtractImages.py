# =========================================
# STEP 2.1: Read .parquet Files and Get Images
# =========================================

'''
Each .parquet file contains documents with embedded images in binary format.
The goal is to:
1. Read all .parquet files from a folder.
2. Combine the data into one DataFrame.
3. Extract and save each image as a separate .png file using its label.
'''
# Used for handling tabular data
import pandas as pd
# Used for reading .parquet files
import pyarrow.parquet as pq
# For file path operaexittions
import os

# Define the folder path where the .parquet files are located
folder_path = r"C:\Users\brian\OneDrive\Escritorio\Skills\Programming\Python\Project"
# List all files ending with .parquet in the folder
parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]

# Initialize an empty list to store individual DataFrames
df_list = []
# Read each .parquet file and append its data to the list
for file in parquet_files:
    full_path = os.path.join(folder_path, file)
    df = pd.read_parquet(full_path) # Load file into DataFrame
    df_list.append(df)

# Combine all individual DataFrames into one large DataFrame
data = pd.concat(df_list, ignore_index=True)
# Preview the columns and first few rows of the dataset
print(data.columns)
print(data.head())


# =========================================
# Extract and Save Images from DataFrame
# =========================================
# To work with images
from PIL import Image
# To handle byte streams
import io

# Define output folder to store extracted .png images
image_output_folder = r"C:\Users\brian\OneDrive\Escritorio\Skills\Programming\Python\Project\extracted_images"
# Create the folder if it doesn't exist
os.makedirs(image_output_folder, exist_ok=True)

# Iterate over each row in the DataFrame to extract and save images
for idx, row in data.iterrows():
    image_bytes = row["image"]["bytes"]     # Extract raw bytes from the 'image' dictionary
    image_id = row["label"]                 # Use the label as part of the image filename

    # Convert bytes to an image object
    image = Image.open(io.BytesIO(image_bytes))

    # Construct a filename and save the image
    image_path = os.path.join(image_output_folder, f"{image_id}_{idx}.png")
    image.save(image_path)
    
# Notify that the images were saved successfully
print("✅ All images have been saved.")