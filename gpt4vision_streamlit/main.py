import streamlit as st
from PIL import Image
from openai import OpenAI
import base64
import requests
import io
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = OpenAI()

# Set your OpenAI API key
client.api_key = os.getenv("OPENAI_API_KEY")

def encode_image_to_base64(image_file):
    # Convert the image file to base64
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def get_openai_vision_response(image_base64, question):
    # Prepare the payload
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 300
    }

    # Make the request to OpenAI
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {client.api_key}"
        },
        json=payload
    )

    return response.json()

def generate_image_with_dalle(prompt):
    try:
        response = client.images.generate(
            model="dall-e-3", # Replace with the correct model name for DALL-E
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response.data[0].url
    except Exception as e:
        return str(e)


st.title("OpenAI Vision and Image Generation")

# Option selection for the user
option = st.radio("Choose an option:", ('Upload Image', 'Generate Image', 'Use Webcam'))

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_to_analyze = uploaded_file.getvalue()

elif option == 'Generate Image':
    dalle_prompt = st.text_input("Enter a prompt to generate an image:", key="dalle_prompt")
    if st.button("Generate Image"):
        generated_image_url = generate_image_with_dalle(dalle_prompt)
        if generated_image_url.startswith("http"):
            st.image(generated_image_url, caption="Generated Image", use_column_width=True)
            img_response = requests.get(generated_image_url)
            image_to_analyze = img_response.content

elif option == 'Use Webcam':
    captured_image = st.camera_input("Take a picture")
    if captured_image:
        image = Image.open(io.BytesIO(captured_image.getvalue()))
        st.image(image, caption="Captured Image", use_column_width=True)
        image_to_analyze = captured_image.getvalue()

# Ask a question about the image
if 'image_to_analyze' in locals():
    question = st.text_input("Ask a question about the image:")
    if st.button("Analyze Image"):
        base64_image = base64.b64encode(image_to_analyze).decode('utf-8')
        response = get_openai_vision_response(base64_image, question)
        st.write("Response from OpenAI Vision API:")
        st.json(response)