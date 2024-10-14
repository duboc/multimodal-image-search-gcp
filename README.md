# GCS Image Uploader, Embedder, and Search

This Streamlit application provides a user-friendly interface for uploading images to Google Cloud Storage (GCS), generating embeddings for these images using Google's Vertex AI MultiModalEmbeddingModel, and performing similarity searches based on text queries or image comparisons.

## Features

1. **Upload Images**: Easily upload a folder of images to a specified Google Cloud Storage bucket.
2. **Generate Embeddings**: Create embeddings for the uploaded images using Google's Vertex AI MultiModalEmbeddingModel.
3. **Visualize Clusters**: Plot image clusters based on their embeddings to visualize similarities.
4. **Text-to-Image Search**: Find images that match a given text description.
5. **Image-to-Image Search**: Find images similar to a selected image.

## Prerequisites

- Python 3.9+
- Google Cloud Platform account with Vertex AI API enabled
- Google Cloud Storage bucket
- Google Cloud SDK installed and initialized
- Docker (for deployment)

## Local Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/gcs-image-search.git
   cd gcs-image-search
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up Google Cloud credentials:
   ```
   gcloud auth application-default login
   ```

## Local Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Use the app:
   - **Upload Tab**: Enter your GCS bucket name and the path to your local image folder, then click "Upload Images".
   - **Embed Tab**: After uploading, click "Generate Embeddings" to create embeddings for the uploaded images.
   - **Search Tab**: 
     - Use "Plot Clusters" to visualize image similarities.
     - Enter a text query and click "Search by Text" to find matching images.
     - Select an image and click "Find Similar Images" to find visually similar images.

## Deployment to Google Cloud Run

1. Update the `deploy.sh` script with your Google Cloud Project ID:
   ```bash
   PROJECT_ID="your-project-id"
   ```

2. Make the deploy script executable:
   ```
   chmod +x deploy.sh
   ```

3. Run the deploy script:
   ```
   ./deploy.sh
   ```

This script will build a Docker image of your application, push it to Google Container Registry, and deploy it to Cloud Run.

## Configuration

- The default GCS bucket is set to "random-file-test". You can change this in the `upload_tab()` function.
- The default local folder path is set to "./backgrounds". Adjust this if your images are in a different location.

## Troubleshooting

- If you encounter authentication issues, make sure your Google Cloud credentials are correctly set up.
- Ensure that your GCS bucket has the necessary permissions set.
- If images fail to load, check that the file formats are supported (JPEG, PNG, etc.).
- For deployment issues, check that you have the necessary APIs enabled in your Google Cloud project.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
