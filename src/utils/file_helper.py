import os 
import shutil
import sys 
import yaml
from pathlib import Path
from src.exception import CustomException
from src.logger import logging


with open("D:\\Langchain Project\\config\\config.yaml") as f:
    config=yaml.safe_load(f)


ALLOWED_EXTENSIONS=config['upload']['allowed_extensions']
UPLOAD_DIR=config['upload']['upload_dir']

MAX_FILE_SIZE_MB = config["upload"].get("max_file_size_mb", 22)
MAX_FILES_COUNT = config["upload"].get("max_files_count", 5)

def validate_file(filename: str) -> bool:
    """
    validates the file extension of the uploaded file.
    returns True if the file extension is allowed, otherwise False.
    """
    try:
        ext=Path(filename).suffix.lower()
        is_valid=ext in ALLOWED_EXTENSIONS
        
        logging.error(f"File extension {ext} is not allowed.")
        return is_valid
    
    except Exception as e:
        logging.error(f"Error validating file: {e}")
        raise CustomException(e, sys) # type: ignore

def validate_file_size(file_bytes:bytes,filename:str)->bool:
    """
    validates the file size of the uploaded file .
    if it's greater than 22 MB then gives error message.
    returns True if the file size is less than or equal to 22 MB, otherwise False
    """
    try:
        size_mb = len(file_bytes) / (1024 * 1024)
        is_valid = size_mb <= MAX_FILE_SIZE_MB
        if not is_valid:
            logging.error(f"File size {size_mb:.2f} MB exceeds the maximum allowed size of {MAX_FILE_SIZE_MB} MB for file {filename}.")
        return is_valid
    except Exception as e:
        logging.error(f"Error validating file size: {e}")
        raise CustomException(e, sys) # type: ignore

def validate_files_count() -> bool:
    """
    Check uploads folder doesn't have too many files
    
    os.listdir() returns list of all files in folder
    if count exceeds limit → reject new upload
    
    example:
        uploads/ has 3 files → 3 < 5 → True  
        uploads/ has 5 files → 5 >= 5 → False 
    """
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        current_count = len(os.listdir(UPLOAD_DIR))
        is_valid = current_count < MAX_FILES_COUNT
        if not is_valid:
            logging.warning(
                f"Upload limit reached: {current_count} files "
                f"(limit: {MAX_FILES_COUNT})"
            )
        return is_valid
    except Exception as e:
        raise CustomException(e, sys)#type:ignore
 
def get_file_extension(filename:str) -> str:
    """
    Gets the extension of a file.
    returns with file extension like .csv, .xlsx, .pdf, .doc etc.
    """
    try:
        return Path(filename).suffix.lower()
    except Exception as e:
        raise CustomException(e, sys) # type: ignore
    
def save_uploaded_file(file_bytes:bytes, filename:str) -> str:
    """
    Saves the uploaded file to the upload directory.
    returns the path of the saved file.
    """
    try:
        if not validate_file(filename):
            raise ValueError(f"File extension not allowed for file: {Path(filename).suffix}. "
                             f"Allowed extensions: {ALLOWED_EXTENSIONS}")
        if not validate_file_size(file_bytes, filename):
            raise ValueError(
                f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
            )
        if not validate_files_count():
            raise ValueError(
                f"Too many files uploaded. "
                f"Maximum {MAX_FILES_COUNT} files at once."
            )
        
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        logging.info(f"File saved successfully: {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error saving file: {e}")
        raise CustomException(e, sys) # type: ignore

def clean_uploads():
    """
    clean previous full memory uploads from the upload directory.
    """
    try:
        if os.path.exists(UPLOAD_DIR):
           shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR,exist_ok=True)
        logging.info("Upload folder cleaned successfully.")
    except Exception as e:
        raise CustomException(e, sys) # type: ignore

def delete_file_after_processing(file_path:str):
    """
    Delete ONE specific file after successful processing
    Better than deleting entire folder

    called ONLY after successful ingestion pipeline
    if pipeline fails → this never called → file stays safe
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"File {file_path} deleted successfully after processing.")

    except Exception as e:
        raise CustomException(e, sys) # type: ignore


def get_filename(file_path:str) -> str:
    """Gets the name of the file from the file path.

    returns the name of the file."""
    return Path(file_path).name

def read_config(config_path:str ="D:\\Langchain Project\\config\\config.yaml")-> dict:
    """ 
    Reads  the configuration file and returns the configuration as a dictionary.
    
    """
    try:
        with open(config_path) as f:
            config=yaml.safe_load(f)
        logging.info(f"Config file read successfully from {config_path}")
        return config
    except Exception as e:
        logging.info(f"Error reading config file: {e}")
        raise CustomException(e, sys) # type: ignore