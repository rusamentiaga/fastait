#!/bin/bash
# download_test_data.sh
# Script to download and extract the test dataset
# https://doi.org/10.17632/v4knrwgj9y.2

set -e  # Exit immediately if a command fails

# Destination directory
DEST_DIR="data"
mkdir -p "$DEST_DIR"

# Download the dataset
ZIP_URL="https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/v4knrwgj9y-2.zip"
ZIP_FILE="${ZIP_URL##*/}"  # Extract filename from URL

echo "Downloading $ZIP_FILE..."
wget -O "$ZIP_FILE" "$ZIP_URL"

# Extract the dataset
echo "Extracting dataset to '$DEST_DIR'..."
unzip -o "$ZIP_FILE" -d "$DEST_DIR"

# Remove the zip file
echo "Cleaning up..."
rm "$ZIP_FILE"

# Recursively unzip any nested zip files
echo "Extracting nested zip files..."
find "$DEST_DIR" -name "*.zip" | while read NESTED_ZIP; do
    unzip -o "$NESTED_ZIP" -d "$(dirname "$NESTED_ZIP")"
    rm "$NESTED_ZIP"
done

echo "Download and extraction completed successfully."