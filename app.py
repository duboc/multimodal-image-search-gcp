import streamlit as st
import os
from google.cloud import storage
from google.auth import default
from vertexai.vision_models import MultiModalEmbeddingModel, Image
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from sklearn.decomposition import PCA
from PIL import Image as PILImage
import requests
from io import BytesIO
import io

# Initialize the model globally
model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")

# Set page configuration to wide mode
st.set_page_config(layout="wide")

def upload_folder_to_gcs(bucket_name, source_folder_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    folder_name = os.path.basename(os.path.normpath(source_folder_path))
    uploaded_files = []

    for root, dirs, files in os.walk(source_folder_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, source_folder_path)
            blob = bucket.blob(relative_path)
            
            # Check if the file already exists in the bucket
            if not blob.exists():
                blob.upload_from_filename(local_file)
                blob.metadata = {'source_folder': folder_name}
                blob.patch()
                st.write(f"Uploaded: {relative_path}")
            else:
                st.write(f"Skipped (already exists): {relative_path}")
            
            uploaded_files.append(f"gs://{bucket_name}/{relative_path}")

    return uploaded_files

def get_image_embeddings(gcs_uri):
    try:
        image = Image.load_from_file(gcs_uri)
        embeddings = model.get_embeddings(
            image=image,
            contextual_text=None,
        )
    except Exception as e:
        st.error(f"Error processing {gcs_uri}: {str(e)}")
        return None
    return embeddings.image_embedding

def process_images(gcs_uris):
    with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers as needed
        futures = [executor.submit(get_image_embeddings, gcs_uri) for gcs_uri in gcs_uris]
        results = [future.result() for future in futures]
    return results

def plot_clusters(df, column):
    # For simplicity, we'll use PCA to reduce dimensions to 2D
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(np.array([v for v in df['vectors'] if v is not None]))
    
    df_plot = pd.DataFrame({
        'x': vectors_2d[:, 0],
        'y': vectors_2d[:, 1],
        'label': df[column].iloc[:len(vectors_2d)]
    })
    
    fig = px.scatter(df_plot, x='x', y='y', color='label', hover_data=['label'])
    st.plotly_chart(fig)

def search(df: pd.DataFrame, image_uri: str=None, query_text: str=None) -> pd.DataFrame:
    df = df.copy()
    if image_uri:
        query_embedding = get_image_embeddings(image_uri)
    elif query_text:
        query_embedding = model.get_embeddings(image=None, contextual_text=query_text).text_embedding
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # Filter out rows with NaN or None values in the 'vectors' column
    df = df.dropna(subset=['vectors'])
    df = df[df['vectors'].apply(lambda x: x is not None)]
    
    # Check if there are any valid vectors left
    if df.empty:
        st.warning("No valid image vectors found. All vectors contain NaN values or are None.")
        return pd.DataFrame()
    
    df['similarity'] = df['vectors'].apply(lambda x: cosine_similarity(query_embedding, np.array(x).reshape(1, -1))[0][0])
    df = df.sort_values(by='similarity', ascending=False).head(20)  # Increase to top 20 results
    
    return df

def upload_tab():
    st.header("Upload Images to GCS")
    bucket_name = st.text_input("Enter the name of your Google Cloud Storage bucket:", value="random-file-test")
    folder_path = st.text_input("Enter the path to the folder containing images:", value="./backgrounds")

    if st.button("Upload Images"):
        if bucket_name and folder_path:
            try:
                credentials, project = default()
                if credentials:
                    uploaded_files = upload_folder_to_gcs(bucket_name, folder_path)
                    st.success(f"Upload completed! {len(uploaded_files)} files were processed.")
                    st.session_state['uploaded_files'] = uploaded_files
                else:
                    st.error("No valid credentials found. Please ensure you have set up application default credentials.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please provide both the bucket name and folder path.")

def embed_tab():
    st.header("Embed Images")
    if 'uploaded_files' not in st.session_state or not st.session_state['uploaded_files']:
        st.warning("Please upload images in the 'Upload' tab first.")
        return

    if st.button("Generate Embeddings"):
        gcs_uris = st.session_state['uploaded_files']
        with st.spinner("Generating embeddings... This may take a while."):
            image_embeddings = process_images(gcs_uris)
        
        df = pd.DataFrame({
            'image': gcs_uris,
            'vectors': image_embeddings,
            'title': [os.path.basename(uri) for uri in gcs_uris]  # Using filename as title
        })
        
        st.session_state['df'] = df
        st.success("Embeddings generated successfully!")
        st.write("Preview of the dataframe:")
        st.dataframe(df.head())
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download embeddings as CSV",
            data=csv,
            file_name="image_embeddings.csv",
            mime="text/csv",
        )

def search_tab():
    st.header("Search and Interact")
    if 'df' not in st.session_state or st.session_state['df'] is None:
        st.warning("Please generate embeddings in the 'Embed' tab first.")
        return

    df = st.session_state['df']
    
    st.subheader("Plot Clusters")
    if st.button("Plot Clusters"):
        df_plot = df.dropna(subset=['vectors'])
        df_plot = df_plot[df_plot['vectors'].apply(lambda x: x is not None)]
        if df_plot.empty:
            st.warning("No valid vectors found for plotting. All vectors contain NaN values or are None.")
        else:
            plot_clusters(df_plot, 'title')

    st.subheader("Text-to-Image Search")
    query_text = st.text_input("Enter your search query:")
    if st.button("Search by Text"):
        if query_text:
            with st.spinner("Searching... Please wait."):
                search_results = search(df, query_text=query_text)
            if search_results.empty:
                st.warning("No valid search results found.")
            else:
                display_search_results(search_results)

    st.subheader("Image-to-Image Search")
    image_uri = st.selectbox("Select an image for similarity search:", df['image'].tolist())
    if st.button("Find Similar Images"):
        if image_uri:
            with st.spinner("Searching for similar images... Please wait."):
                search_results = search(df, image_uri=image_uri)
            if search_results.empty:
                st.warning("No valid similar images found.")

def display_search_results(df):
    st.write("### Search Results")
    
    # Display top 6 images as thumbnails in the first row
    st.write("Top Results:")
    cols = st.columns(6)
    for index, (_, row) in enumerate(df.head(6).iterrows()):
        with cols[index]:
            display_thumbnail(row)
    
    # Display detailed results
    st.write("Detailed Results:")
    for index, row in df.iterrows():
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                display_thumbnail(row, size=(150, 150))
            with col2:
                st.write(f"**{row['title']}**")
                st.write(f"Similarity: {row['similarity']:.4f}")
                st.write("---")

def display_thumbnail(row, size=(100, 100)):
    try:
        gcs_uri = row['image']
        bucket_name = gcs_uri.split('/')[2]
        blob_name = '/'.join(gcs_uri.split('/')[3:])
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        image_content = blob.download_as_bytes()
        img = PILImage.open(io.BytesIO(image_content))
        img.thumbnail(size)
        
        st.image(img, use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

def main():
    st.title("GCS Image Uploader, Embedder, and Search")

    tab1, tab2, tab3 = st.tabs(["Upload", "Embed", "Search"])

    with tab1:
        upload_tab()

    with tab2:
        embed_tab()

    with tab3:
        search_tab()

if __name__ == "__main__":
    main()
