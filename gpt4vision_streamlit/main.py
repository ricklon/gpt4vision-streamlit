import streamlit as st
from PIL import Image
from openai import OpenAI
import base64
import requests
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
# Image Analysis Section
st.header("Image Analysis with GPT-4 Vision")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
question = st.text_input("Ask a question about the image:")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    if st.button("Analyze Uploaded Image"):
        base64_image = encode_image_to_base64(uploaded_file)
        response = get_openai_vision_response(base64_image, question)
        st.write("Response from OpenAI Vision API:")
        st.json(response)

# Image Generation Section
st.header("Generate Image with DALL-E")
dalle_prompt = st.text_input("Enter a prompt to generate an image:", key="dalle_prompt")
if st.button("Generate Image"):
    generated_image_url = generate_image_with_dalle(dalle_prompt)
    if generated_image_url.startswith("http"):
        st.image(generated_image_url, caption="Generated Image", use_column_width=True)
        st.session_state['generated_image_url'] = generated_image_url

        # Download Button for the image
        try:
            img_response = requests.get(generated_image_url)
            img = Image.open(BytesIO(img_response.content))
            buf = BytesIO()
            img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(label="Download Image",
                            data=byte_im,
                            file_name="generated_image.png",
                            mime="image/png")
        except Exception as e:
            st.error("Error in downloading the image: " + str(e))

# Option to analyze the generated image
if 'generated_image_url' in st.session_state:
    if st.button("Analyze Generated Image"):
        response = get_openai_vision_response(st.session_state['generated_image_url'], question)
        st.write("Response from OpenAI Vision API:")
        st.json(response)