


import io
import os
import string
import random
import requests
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

def get_google_font(font_name, font_size):
    url = f"https://fonts.googleapis.com/css?family={font_name}"
    response = requests.get(url)
    css = response.text
    font_url = css.split("url(")[1].split(")")[0]
    font_response = requests.get(font_url)
    return ImageFont.truetype(io.BytesIO(font_response.content), font_size)

def generate_text_image(text, font_name, font_size, background_color='white', text_color='black'):
    font = get_google_font(font_name, font_size)
    dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    left, top, right, bottom = dummy_draw.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top
    image = Image.new('RGB', (text_width, text_height+7), color=background_color)
    draw = ImageDraw.Draw(image)
    draw.text((0, -5), text, font=font, fill=text_color)
    return image

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))

def generate_and_save_images(num_images, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fonts = ["Roboto", "Open+Sans", "Lato", "Montserrat", "Raleway"]
    
    for i in tqdm(range(num_images)):
        text = list(str(random.randint(0, 999999)) + generate_random_string(5))
        random.shuffle(text)
        text = ''.join(text)
        text = '0'*random.randint(2,7) + text
        font = random.choice(fonts)
        font_size = random.randint(24, 48)
        text_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
        image = generate_text_image(text, font, font_size, 'white', text_color)
        filename = f"{save_dir}/{text}.png"
        image.save(filename)



# Generate and save 1000 images
generate_and_save_images(1000, 'generated_images')



