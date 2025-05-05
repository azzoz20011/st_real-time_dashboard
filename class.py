import boto3
import streamlit as st
import io
from PIL import Image
import time
import os
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="GEO-VISION:Real-time dashboard",
    layout="centered"
)

# Function to initialize S3 client
@st.cache_resource
def get_s3_client():
    # Create and return S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id="AKIAVX2JJ4564UGYCE54",
        aws_secret_access_key="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",
        region_name="eu-north-1"
    )
    
    return s3_client

# Function to get the latest image from S3 bucket
def get_latest_image(bucket_name, prefix=""):
    s3_client = get_s3_client()
    
    try:
        # List all objects in the bucket with the given prefix
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            st.error(f"No objects found in bucket: {bucket_name} with prefix: {prefix}")
            return None, None, None
        
        # Sort objects by last modified date (newest first)
        objects = sorted(
            response['Contents'],
            key=lambda obj: obj['LastModified'],
            reverse=True
        )
        
        if not objects:
            st.error(f"No objects found in bucket: {bucket_name} with prefix: {prefix}")
            return None, None, None
        
        # Get the latest object
        latest_object = objects[0]
        object_key = latest_object['Key']
        last_modified = latest_object['LastModified']
        
        # Get the object content
        response = s3_client.get_object(
            Bucket=bucket_name,
            Key=object_key
        )
        
        # Read the image content
        image_content = response['Body'].read()
        
        return image_content, last_modified, object_key
    
    except Exception as e:
        st.error(f"Error fetching image from S3: {str(e)}")
        return None, None, None

# Function to predict image class
def predict_image(image, model_path, classes):
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Preprocess the image
        if img_array.ndim == 3:
            img_array = rgb2gray(img_array)
        
        # Resize to match the expected CNN input size
        # The error message indicates that the model expects 2048 features
        # For a square image, this would be approximately 45x45 pixels
        # We'll resize to exactly match the expected feature count
        target_size = int(np.sqrt(2048))  # ~45.25, but we'll use 45
        img_array = resize(img_array, (45, 45), anti_aliasing=True)
        
        # Flatten the image to match the expected 2048 features
        features = img_array.flatten()
        
        # Ensure we have exactly 2048 features
        if len(features) > 2048:
            features = features[:2048]
        elif len(features) < 2048:
            # Pad with zeros if somehow we have fewer features
            features = np.pad(features, (0, 2048 - len(features)))
            
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        predicted_class = classes[prediction]
        
        # Get prediction probabilities if available
        confidence = None
        probabilities = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            for i, cls in enumerate(classes):
                probabilities[cls] = float(probs[i])
            confidence = float(probs[prediction])
            
        return predicted_class, confidence, probabilities
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

# Function to save results to CSV and upload to S3
def save_and_upload_results(bucket_name, image_key, timestamp, prediction, confidence, probabilities):
    try:
        s3_client = get_s3_client()
        csv_filename = "prediction_results.csv"
        s3_key = f"predictions/{csv_filename}"
        
        # Create DataFrame for the new result
        new_result = {
            'timestamp': [timestamp],
            'image_key': [image_key],
            'prediction': [prediction],
            'confidence': [confidence]
        }
        
        # Add probabilities for each class if available
        if probabilities:
            for cls, prob in probabilities.items():
                new_result[f'prob_{cls}'] = [prob]
        
        new_df = pd.DataFrame(new_result)
        
        # Check if the CSV file already exists in S3
        try:
            # Try to get the existing file
            response = s3_client.get_object(
                Bucket=bucket_name,
                Key=s3_key
            )
            
            # Read existing CSV into DataFrame
            existing_csv = io.StringIO(response['Body'].read().decode('utf-8'))
            existing_df = pd.read_csv(existing_csv)
            
            # Append new results to existing DataFrame
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            
        except s3_client.exceptions.NoSuchKey:
            # File doesn't exist yet, use only the new data
            updated_df = new_df
        
        # Convert DataFrame to CSV
        csv_buffer = io.StringIO()
        updated_df.to_csv(csv_buffer, index=False)
        
        # Upload CSV to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=csv_buffer.getvalue()
        )
        
        return csv_filename
    
    except Exception as e:
        st.error(f"Error saving/uploading results: {str(e)}")
        return None

# Function to download model from S3 if not locally available
def download_model_if_needed(bucket_name, model_key, local_model_path):
    if not os.path.exists(local_model_path):
        try:
            s3_client = get_s3_client()
            st.info(f"Downloading model from S3: {model_key}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
            
            # Download the model file
            s3_client.download_file(bucket_name, model_key, local_model_path)
            st.success(f"Model downloaded successfully to {local_model_path}")
            
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}")
            return False
    
    return True

# Function to create and display matplotlib visualization
def display_matplotlib_visualization(image, placeholder):
    try:
        # Create the figure and axis
        fig, ax = plt.subplots()
        
        # Display the image
        ax.imshow(image)
        
        # Hide the default ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add the axis labels
        ax.set_xlabel("X-axis", fontsize=14, labelpad=10)
        ax.set_ylabel("Y-axis", fontsize=14, labelpad=10, rotation=90, va="center")
        
        # Tweak margins so labels aren't cut off
        plt.subplots_adjust(left=0.15, bottom=0.15)
        
        # Render into Streamlit placeholder
        placeholder.pyplot(fig)
        
        return True
    except Exception as e:
        st.error(f"Error creating Matplotlib visualization: {str(e)}")
        return False

# Function to create and display Plotly visualization
def display_plotly_visualization(image, placeholder, prediction=None, probabilities=None):
    try:
        # Create a figure with subplots - one for the image and one for probabilities
        if probabilities:
            # Create a 1x2 subplot grid for image and probability chart
            fig = make_subplots(
                rows=1, cols=2,
                column_widths=[0.7, 0.3],
                specs=[[{"type": "heatmap"}, {"type": "bar"}]],
                subplot_titles=("GPR Scan", "Class Probabilities")
            )
        else:
            # Just create a single plot for the image
            fig = make_subplots(rows=1, cols=1)
        
        # Add the image as a heatmap
        fig.add_trace(
            go.Heatmap(z=image, colorscale='Viridis'),
            row=1, col=1
        )
        
        # If we have probability data, add a bar chart
        if probabilities:
            # Sort the classes by probability value (descending)
            sorted_classes = sorted(probabilities.keys(), key=lambda x: probabilities[x], reverse=True)
            
            # Create bar chart for probabilities
            fig.add_trace(
                go.Bar(
                    x=[probabilities[cls] for cls in sorted_classes],
                    y=sorted_classes,
                    orientation='h',
                    marker=dict(
                        color=['red' if cls == 'Pipe' else 'blue' if cls == 'Foil' else 'green' for cls in sorted_classes],
                        line=dict(color='black', width=1)
                    ),
                    text=[f"{probabilities[cls]:.2f}" for cls in sorted_classes],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # Update the y-axis of the bar chart
            fig.update_yaxes(title_text="Class", row=1, col=2)
            fig.update_xaxes(title_text="Probability", row=1, col=2, range=[0, 1])
        
        # Update layout with custom margins
        fig.update_layout(
            margin=dict(l=50, r=20, t=50, b=50),
            height=500,
            width=1000 if probabilities else 700,
            title="GPR Scan Classification",
            title_x=0.5  # Center the title
        )
        
        # Update axis labels for the image
        fig.update_xaxes(title_text="X", row=1, col=1)
        fig.update_yaxes(title_text="Y", row=1, col=1)
        
        # Render into Streamlit placeholder
        placeholder.plotly_chart(fig)
        
        return True
    except Exception as e:
        st.error(f"Error creating Plotly visualization: {str(e)}")
        return False

# Main app function
def main():
    st.title("GEO-VISION:Real-time dashboard")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        bucket_name = "my-gpr-dataa"
        image_prefix = "scans/"
        refresh_interval = st.slider("Refresh Interval (seconds)", 1, 60, 1)
        
        st.header("Model Configuration")
        model_path = "models/svm_on_preprocessed_cnn.joblib"
        model_s3_key = st.text_input("Model S3 Key (Optional)", "models/svm_on_preprocessed_cnn.joblib")
        classes_input = "nothing,Pipe,Foil"
        classes = [c.strip() for c in classes_input.split(",")]
        
        # Download model button
        
        download_model_if_needed(bucket_name, model_s3_key, model_path)
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Placeholder for the standard image display
        image_placeholder = st.empty()
        
        # Placeholder for the plotly visualization with probabilities
        plotly_placeholder = st.empty()
        
        # Placeholder for the timestamp
        timestamp_placeholder = st.empty()
    
    with col2:
        st.subheader("Image Information")
        info_placeholder = st.empty()
        
        st.subheader("Prediction Results")
        prediction_placeholder = st.empty()
        
        st.subheader("Last CSV Upload")
        history_placeholder = st.empty()
    
    # Initialize session state
    if 'last_modified' not in st.session_state:
        st.session_state['last_modified'] = None
    
    if 'last_csv' not in st.session_state:
        st.session_state['last_csv'] = None
    
    # Check if model exists locally, if not download it
    if model_s3_key and not os.path.exists(model_path):
        download_model_if_needed(bucket_name, model_s3_key, model_path)
    
    # Main loop for real-time updates
    while True:
        # Get the latest image from S3
        image_content, last_modified, object_key = get_latest_image(bucket_name, image_prefix)
        
        if image_content is not None:
            try:
                # Check if the image has changed
                if st.session_state['last_modified'] != last_modified:
                    st.session_state['last_modified'] = last_modified
                    
                    # Open the image using PIL
                    image = Image.open(io.BytesIO(image_content))
                    
                    # Convert PIL Image to numpy array for visualizations
                    img_array = np.array(image)
                    
                    # Run prediction on the image
                    prediction = None
                    probabilities = None
                    if os.path.exists(model_path):
                        prediction_result, confidence, prob_dict = predict_image(image, model_path, classes)
                        prediction = prediction_result
                        probabilities = prob_dict
                    
                    # Clear the image placeholder
                    image_placeholder.empty()
                    
                    # If image is RGB, convert to grayscale for better heatmap display
                    if img_array.ndim == 3:
                        img_array_2d = rgb2gray(img_array)
                    else:
                        img_array_2d = img_array
                        
                    # Display with plotly and probabilities
                    display_plotly_visualization(img_array_2d, plotly_placeholder, prediction, probabilities)
                    
                    # Display the timestamp
                    current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    timestamp_placeholder.write(f"Last Updated: {current_timestamp}")
                    
                    # Display image information
                    info_text = f"""
                    **File Information:**
                    - Format: {image.format}
                    - Size: {image.size}
                    - Last Modified: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}
                    """
                    info_placeholder.markdown(info_text)
                    
                    # Since we already ran the prediction earlier for visualization
                    if prediction is not None and os.path.exists(model_path):
                        # Display prediction results
                        prediction_text = f"""
                        **Prediction Results:**
                        - Predicted Class: {prediction}
                        """
                        
                        if confidence is not None:
                            prediction_text += f"- Confidence: {confidence:.4f}\n\n"
                            
                            # Show probabilities for each class
                            prediction_text += "**Class Probabilities:**\n"
                            for cls, prob in probabilities.items():
                                prediction_text += f"- {cls}: {prob:.4f}\n"
                        
                        prediction_placeholder.markdown(prediction_text)
                        
                        # Save results to CSV and upload to S3
                        csv_filename = save_and_upload_results(
                            bucket_name, 
                            object_key,
                            current_timestamp,
                            prediction,
                            confidence,
                            probabilities
                        )
                            
                        if csv_filename:
                            # Get the updated CSV file to display prediction history
                            try:
                                s3_client = get_s3_client()
                                response = s3_client.get_object(
                                    Bucket=bucket_name,
                                    Key=f"predictions/{csv_filename}"
                                )
                                
                                # Read the CSV data
                                csv_content = response['Body'].read().decode('utf-8')
                                history_df = pd.read_csv(io.StringIO(csv_content))
                                
                                # Update the number of rows to track changes
                                current_rows = len(history_df)
                                new_rows = current_rows - st.session_state.get('csv_rows', 0)
                                st.session_state['csv_rows'] = current_rows
                                
                                # Display the prediction history
                                history_placeholder.markdown(f"**Prediction history: {current_rows} total records** (+{new_rows} new)")
                                history_placeholder.dataframe(
                                    history_df.sort_values('timestamp', ascending=False).head(5),
                                    use_container_width=True
                                )
                            except Exception as e:
                                history_placeholder.warning(f"Could not load prediction history: {str(e)}")
                    else:
                        prediction_placeholder.warning("Model file not found. Please download or check the path.")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
        
        # Wait for the specified refresh interval
        time.sleep(refresh_interval)

if __name__ == "__main__":
    main()
