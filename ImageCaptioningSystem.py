import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
from gtts import gTTS
import base64
import os
import google.generativeai as genai
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import zipfile
import re
import textwrap

# Set page configuration
st.set_page_config(page_title="Image Captioning System", layout="wide")

# --- Configuration Section ---
# IMPORTANT: Replace these with your own API keys or use environment variables
# Configure Gemini API (Get your API key from: https://ai.google.dev/)
# genai.configure(api_key="YOUR_GEMINI_API_KEY")  # Uncomment and add your key

# Alternatively, use environment variables:
# import os
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Model Loading ---
@st.cache_resource
def load_blip_model():
    """Load BLIP model for initial image captioning"""
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Convert a given caption text to speech and return audio as a byte stream
def generate_audio(text):
    try:
        # Remove special characters that might cause issues with TTS
        clean_text = ''.join(c for c in text if c.isalnum() or c.isspace() or c in ',.!?-:;\'\"')
        tts = gTTS(clean_text)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        # Return a minimal valid audio file in case of error
        return None

# Optimized prompt formats for advanced image captioning with error-handling
caption_styles = {
    "Generalised Caption": (
        "Analyze the image and generate a concise, general-purpose caption that broadly describes the scene or subject. "
        "Use simple, clear language suitable for all audiences. "
        "If the image is unclear or ambiguous, describe what is most prominent without guessing."
    ),

    "Creative Caption": (
        "Observe the image and create a highly imaginative and artistic caption. Use metaphor, personification, or poetic language. "
        "Make it emotionally engaging and original. If the content is vague or abstract, focus on evoking a feeling or theme rather than specifics."
    ),

    "Professional Caption": (
        "Generate a formal, polished caption suitable for professional or business use. Emphasize clarity, purpose, and tone appropriate for marketing, branding, or corporate presentations. "
        "If the image lacks context, focus on neutral, safe interpretations that could fit a wide professional setting."
    ),

    "Descriptive Caption": (
        "Examine the image and write a descriptive caption of about 25 words. Focus on visually important elements like objects, setting, actions, and mood. "
        "Avoid interpretation or assumption—stick to visible, factual details. If the image is blurry or unclear, describe what can be confidently identified."
    ),

    "Quote Caption": (
        "Interpret the essence or theme of the image and generate an inspiring or thoughtful quote that matches the visual. "
        "Attribute it to a relevant or fictional person (e.g., a philosopher, artist, or visionary). "
        "If the image content is not meaningful enough for a quote, provide a general life reflection loosely inspired by it."
    )
}


# Use BLIP to generate a base caption
def generate_blip_caption(image):
    processor, model = load_blip_model()
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    return processor.batch_decode(output, skip_special_tokens=True)[0]

# Use Gemini to generate multiple stylized captions based on base BLIP caption
def generate_gemini_captions(image, base_caption, style, count=5):
    # Convert PIL image to bytes for Gemini
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    # Create a multimodal prompt with both the image and text
    model = genai.GenerativeModel('gemini-1.5-pro')
    prompt = f"{caption_styles[style]}\nBase description: {base_caption}\nGenerate {count} unique, high-quality captions."
    
    response = model.generate_content([
        prompt,
        {"mime_type": "image/jpeg", "data": img_byte_arr.getvalue()}
    ])
    
    # Process the response to extract captions
    captions = []
    for line in response.text.strip().split('\n'):
        # Clean up numbering, bullets, etc.
        clean_line = line.strip()
        if clean_line and not clean_line.isdigit():
            # Remove numbering patterns like "1.", "1)", "[1]", etc.
            import re
            clean_line = re.sub(r'^(\d+[\.\)\]]\s*|\*\s*|\-\s*)', '', clean_line)
            captions.append(clean_line.strip())
    
    # Return only the requested number of captions
    return captions[:count]

# Parse markdown-like formatting for captions
def parse_caption_formatting(caption):
    # Store original caption for audio generation
    original_caption = caption
    
    # Track formatting for text segments
    formatted_segments = []
    
    # Parse bold text (**text**)
    bold_pattern = r'\*\*(.*?)\*\*'
    bold_matches = re.finditer(bold_pattern, caption)
    
    for match in bold_matches:
        start, end = match.span()
        formatted_segments.append({
            'text': match.group(1),
            'start': start,
            'end': end,
            'bold': True,
            'italic': False,
            'underline': False
        })
    
    # Parse italic text (*text*)
    italic_pattern = r'(?<!\*)\*([^\*]+)\*(?!\*)'
    italic_matches = re.finditer(italic_pattern, caption)
    
    for match in italic_matches:
        start, end = match.span()
        # Check if this segment overlaps with a bold segment
        is_new_segment = True
        for segment in formatted_segments:
            if (start >= segment['start'] and start < segment['end']) or \
               (end > segment['start'] and end <= segment['end']):
                segment['italic'] = True
                is_new_segment = False
                break
        
        if is_new_segment:
            formatted_segments.append({
                'text': match.group(1),
                'start': start,
                'end': end,
                'bold': False,
                'italic': True,
                'underline': False
            })
    
    # Parse underlined text (_text_)
    underline_pattern = r'_(.*?)_'
    underline_matches = re.finditer(underline_pattern, caption)
    
    for match in underline_matches:
        start, end = match.span()
        # Check if this segment overlaps with an existing segment
        is_new_segment = True
        for segment in formatted_segments:
            if (start >= segment['start'] and start < segment['end']) or \
               (end > segment['start'] and end <= segment['end']):
                segment['underline'] = True
                is_new_segment = False
                break
        
        if is_new_segment:
            formatted_segments.append({
                'text': match.group(1),
                'start': start,
                'end': end,
                'bold': False,
                'italic': False,
                'underline': True
            })
    
    # Clean up the caption by removing markdown syntax
    clean_caption = re.sub(r'\*\*(.*?)\*\*', r'\1', caption)  # Remove bold markers
    clean_caption = re.sub(r'(?<!\*)\*([^\*]+)\*(?!\*)', r'\1', clean_caption)  # Remove italic markers
    clean_caption = re.sub(r'_(.*?)_', r'\1', clean_caption)  # Remove underline markers
    
    return {
        'original': original_caption,
        'clean': clean_caption,
        'segments': sorted(formatted_segments, key=lambda x: x['start'])
    }

# Overlay selected caption text on top of the image with formatting support
def add_caption_to_image(image, caption):
    width, height = image.size
    
    # Parse caption formatting
    caption_data = parse_caption_formatting(caption)
    clean_caption = caption_data['clean']
    
    # Dynamically adjust font size based on image dimensions - bigger font sizes
    font_size = max(32, min(64, int(width/14)))  # Even larger font sizes for better visibility
    
    # Prepare font options - modern fonts with better visibility
    font_families = [
        # Primary choice - Calibri (modern, clean font)
        ("Calibri", "Calibri Bold", "Calibri Italic", "Calibri Bold Italic"),
        # Second choice - Segoe UI (modern Microsoft font)
        ("Segoe UI", "Segoe UI Bold", "Segoe UI Italic", "Segoe UI Bold Italic"),
        # Third choice - Arial (universally available)
        ("Arial", "Arial Bold", "Arial Italic", "Arial Bold Italic"),
        # Fourth choice - specific filenames
        ("calibri.ttf", "calibrib.ttf", "calibrii.ttf", "calibriz.ttf"),
        ("segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "segoeuiz.ttf"),
        ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")
    ]
    
    # Try each font family until one works
    regular_font = None
    bold_font = None
    italic_font = None
    bold_italic_font = None
    
    for font_family in font_families:
        try:
            regular_font = ImageFont.truetype(font_family[0], font_size)
            bold_font = ImageFont.truetype(font_family[1], font_size)
            italic_font = ImageFont.truetype(font_family[2], font_size)
            bold_italic_font = ImageFont.truetype(font_family[3], font_size)
            break  # If we successfully loaded all fonts, break the loop
        except:
            continue
    
    # If no fonts were loaded successfully, use default font with artificial styles
    if regular_font is None:
        regular_font = ImageFont.load_default()
        bold_font = regular_font
        italic_font = regular_font
        bold_italic_font = regular_font
    
    # Use smaller margins to allow for larger text
    max_width = width - 80  # Slightly larger margin for better appearance
    
    # Better character width estimation for wrapping
    wrapper = textwrap.TextWrapper(width=max(1, int(max_width / (font_size * 0.45))))  # Even more space for text
    wrapped_lines = wrapper.wrap(clean_caption)
    
    # Add extra space if formatting is detected
    formatting_detected = len(caption_data['segments']) > 0
    
    # Calculate required height for caption area
    line_height = font_size * 1.8  # Increased line spacing
    padding = 120 if formatting_detected else 100  # More padding for better visual appearance
    bar_height = int(len(wrapped_lines) * line_height + padding)
    
    # Create new image with appropriate white bar at top
    result_img = Image.new("RGB", (width, height + bar_height), (255, 255, 255))  # Pure white background
    result_img.paste(image, (0, bar_height))
    
    # Draw on the image
    draw = ImageDraw.Draw(result_img)
    
    # Draw each line of text with proper formatting
    y_position = 60  # Increased top margin for better spacing
    
    # If no formatting is detected but markdown characters are present,
    # treat the entire text as formatted based on simple rules
    if not formatting_detected and ('**' in caption or '*' in caption or '_' in caption):
        is_bold = '**' in caption
        is_italic = '*' in caption and not '**' in caption
        is_underline = '_' in caption
        
        # Assign the right font based on detected formatting
        if is_bold and is_italic:
            font = bold_italic_font
        elif is_bold:
            font = bold_font
        elif is_italic:
            font = italic_font
        else:
            font = regular_font
            
        # Draw each line with the detected formatting
        for line in wrapped_lines:
            # Center the text
            line_width = font.getlength(line) if hasattr(font, 'getlength') else font.getsize(line)[0]
            x_position = (width - line_width) // 2
            
            # Draw shadow effect for better visibility
            shadow_color = (220, 220, 220)  # Light gray shadow
            draw.text((x_position + 2, y_position + 2), line, fill=shadow_color, font=font)
            
            # Draw the main text
            if is_bold and font == regular_font:  # If we're using fallback and need bold
                for offset in range(-1, 2):
                    draw.text((x_position + offset, y_position), line, fill="black", font=font)
                    draw.text((x_position, y_position + offset), line, fill="black", font=font)
            else:
                draw.text((x_position, y_position), line, fill="black", font=font)
            
            # Draw underline if needed
            if is_underline:
                line_width = font.getlength(line) if hasattr(font, 'getlength') else font.getsize(line)[0]
                underline_y = y_position + font_size + 3
                draw.line((x_position, underline_y, x_position + line_width, underline_y), fill="black", width=3)
                
            y_position += line_height
    else:
        # Process line by line with individual segment formatting
        for line in wrapped_lines:
            # Calculate center position for this line
            line_width = regular_font.getlength(line) if hasattr(regular_font, 'getlength') else regular_font.getsize(line)[0]
            x_position = (width - line_width) // 2
            
            # Draw shadow for the base text
            draw.text((x_position + 2, y_position + 2), line, fill=(220, 220, 220), font=regular_font)
            
            # Draw the base text
            draw.text((x_position, y_position), line, fill="black", font=regular_font)
            
            # Then overlay formatted segments if we have any
            if caption_data['segments']:
                current_pos = 0
                for segment in caption_data['segments']:
                    if segment['text'] in line:
                        segment_start = line.find(segment['text'], current_pos)
                        if segment_start != -1:
                            # Calculate position for this segment
                            before_text = line[:segment_start]
                            before_width = regular_font.getlength(before_text) if hasattr(regular_font, 'getlength') else regular_font.getsize(before_text)[0]
                            segment_x = x_position + before_width
                            
                            # Select appropriate font based on formatting
                            if segment['bold'] and segment['italic']:
                                font = bold_italic_font
                            elif segment['bold']:
                                font = bold_font
                            elif segment['italic']:
                                font = italic_font
                            else:
                                font = regular_font
                            
                            # Draw the formatted text with shadow and background
                            text_height = font_size
                            text_width = font.getlength(segment['text']) if hasattr(font, 'getlength') else font.getsize(segment['text'])[0]
                            
                            # Draw white background
                            draw.rectangle([segment_x-2, y_position-2, segment_x+text_width+2, y_position+text_height+2], 
                                         fill=(255, 255, 255))
                            
                            # Draw shadow
                            draw.text((segment_x + 2, y_position + 2), segment['text'], 
                                    fill=(220, 220, 220), font=font)
                            
                            # Draw the formatted text
                            if segment['bold'] and font == regular_font:  # If we're using fallback and need bold
                                for offset in range(-1, 2):
                                    draw.text((segment_x + offset, y_position), segment['text'], 
                                            fill="black", font=font)
                                    draw.text((segment_x, y_position + offset), segment['text'], 
                                            fill="black", font=font)
                            else:
                                draw.text((segment_x, y_position), segment['text'], 
                                        fill="black", font=font)
                            
                            # Draw underline if needed - thicker line for better visibility
                            if segment['underline']:
                                underline_y = y_position + font_size + 3
                                draw.line((segment_x, underline_y, segment_x + text_width, underline_y), 
                                        fill="black", width=3)
                            
                            current_pos = segment_start + len(segment['text'])
            
            y_position += line_height
    
    return result_img

# --- Streamlit App UI ---
st.title("🖼️ Advanced Image Captioning System")
st.write("Upload an image and generate creative captions with audio!")

# Initialize session state for tracking the selected style
if 'current_style' not in st.session_state:
    st.session_state.current_style = None

# Step 1: Upload an image
uploaded_file = st.file_uploader("Step 1: Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Step 2: Select caption style from dropdown
    st.write("Step 2: Select the type of caption to generate")
    selected_style = st.selectbox("Caption Style:", list(caption_styles.keys()))
    
    # Check if style has changed
    if st.session_state.current_style != selected_style:
        # Clear previous captions when style changes
        if 'styled_captions' in st.session_state:
            st.session_state.styled_captions = []
        # Update the current style
        st.session_state.current_style = selected_style
    
    # Initialize session state for storing captions
    if 'styled_captions' not in st.session_state:
        st.session_state.styled_captions = []
    
    # Generate captions button
    if st.button("Generate Captions") or ('styled_captions' in st.session_state and len(st.session_state.styled_captions) > 0):
        # Only run the generation if we don't already have captions
        if len(st.session_state.styled_captions) == 0:
            with st.spinner("Analyzing image and generating captions..."):
                # Generate a base caption using BLIP model
                base_caption = generate_blip_caption(image)
                
                # Generate five stylized captions using Gemini
                styled_captions = generate_gemini_captions(image, base_caption, selected_style)
                st.session_state.styled_captions = styled_captions
        
        # Display the captions with audio buttons
        st.subheader("Top 5 Generated Captions:")
        st.write("Each caption has an audio button and a generate image button.")
         # Add some CSS for better styling
        st.markdown("""
        <style>
            .caption-container {
                background-color: grey;
                border-radius: 10px;
                padding: 1px;
                margin-bottom: 15px;
                border: 1px solid #dee2e6;
            }
        </style>
        """, unsafe_allow_html=True)
        
        for i, caption in enumerate(st.session_state.styled_captions):
            # Create a container for each caption
            #st.markdown(f"""<div class="caption-container"></div>""", unsafe_allow_html=True)
            st.markdown("---")
            
            # Create three columns: caption, audio button, and generate button
            col1, col2, col3 = st.columns([6, 1, 3])
            
            with col1:
                # Display the caption text
                st.markdown(f"**Caption {i+1}:**")
                st.write(caption)
            
            with col2:
                # Generate and display audio button
                st.markdown("**Audio:**")
                audio = generate_audio(caption)
                if audio is not None:
                    st.audio(audio, format="audio/mp3")
                else:
                    st.warning("❌")
            
            with col3:
                # Add a dedicated button for generating the image with this caption
                st.markdown("**Generate:**")
                if st.button(f"Generate Image", key=f"generate_btn_{i}"):
                    # Store the selected caption and generate the image
                    st.session_state.selected_caption = caption
                    # Store the raw caption as well for audio generation
                    st.session_state.raw_caption = caption
                    st.session_state.final_image = add_caption_to_image(image, caption)
                    # Set a flag to indicate that we should display the image
                    st.session_state.show_final_image = True
                    # Store the index of the selected caption
                    st.session_state.selected_caption_index = i
        
        # Display the final image if it has been generated
        if 'show_final_image' in st.session_state and st.session_state.show_final_image and 'final_image' in st.session_state:
            st.markdown("---")
            st.subheader("Final Image with Caption:")
            
            # Display which caption was selected
            if 'selected_caption_index' in st.session_state:
                st.info(f"Using Caption {st.session_state.selected_caption_index + 1}: {st.session_state.selected_caption}")
            
            st.image(st.session_state.final_image, use_column_width=True)
            
            # Generate audio for the selected caption if not already generated
            if 'selected_caption_audio' not in st.session_state and 'raw_caption' in st.session_state:
                # Use the raw caption without formatting for better audio
                clean_caption = parse_caption_formatting(st.session_state.raw_caption)['clean']
                st.session_state.selected_caption_audio = generate_audio(clean_caption)
            
            # Create a ZIP file containing both the image and audio
            if 'selected_caption_audio' in st.session_state:
                # Convert final image to bytes
                img_bytes = io.BytesIO()
                st.session_state.final_image.save(img_bytes, format="JPEG")
                img_bytes.seek(0)
                
                # Create a zip file in memory
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    # Add the image to the zip
                    zip_file.writestr('captioned_image.jpg', img_bytes.getvalue())
                    
                    # Add the audio to the zip
                    if st.session_state.selected_caption_audio is not None:
                        audio_bytes = st.session_state.selected_caption_audio.getvalue()
                        zip_file.writestr('caption_audio.mp3', audio_bytes)
                
                # Reset buffer position
                zip_buffer.seek(0)
                
                # Add a download button for the zip file
                st.download_button(
                    "📥 Download Image & Audio (ZIP)", 
                    zip_buffer, 
                    file_name="captioned_image_with_audio.zip", 
                    mime="application/zip",
                    use_container_width=False
                )
                
                # Also offer individual downloads
                st.download_button(
                    "📥 Download Image Only", 
                    img_bytes, 
                    file_name="captioned_image.jpg", 
                    mime="image/jpeg",
                    use_container_width=False
                )
else:
    st.info("Please upload an image to get started.") 
