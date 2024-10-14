import os
from google.cloud import storage

def upload_folder_to_gcs(bucket_name, source_folder_path):
    # Initialize the Google Cloud Storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Get the folder name from the source path
    folder_name = os.path.basename(os.path.normpath(source_folder_path))

    # Walk through the folder
    for root, dirs, files in os.walk(source_folder_path):
        for file in files:
            # Get the full local path of the file
            local_file = os.path.join(root, file)

            # Get the relative path from the source folder
            relative_path = os.path.relpath(local_file, source_folder_path)

            # Create a blob with the same relative path in GCS
            blob = bucket.blob(relative_path)

            # Upload the file
            blob.upload_from_filename(local_file)

            # Set metadata
            blob.metadata = {'source_folder': folder_name}
            blob.patch()

            print(f"File {local_file} uploaded to {relative_path} with metadata: {blob.metadata}")

# Usage example
if __name__ == "__main__":
    bucket_name = "random-file-test"
    source_folder_path = "/Users/duboc/local/backgrounds"
    upload_folder_to_gcs(bucket_name, source_folder_path)