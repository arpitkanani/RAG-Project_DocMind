# Utility Documentation: `src/utils/file_helper.py`

## 1. Overview & Purpose
`src/utils/file_helper.py` provides filesystem operations, file format validation, upload size verification, safe temporary file storage, and cleanup routines for DocuVortex.

---

## 2. Key Components & Functions

### `validate_file(filename: str) -> bool`
- Inspects the file suffix and verifies it exists in `config["upload"]["allowed_extensions"]` (`.pdf`, `.txt`, `.docx`, `.csv`, `.md`, `.xlsx`). Case-insensitive.

### `validate_file_size(file_bytes: bytes, filename: str) -> bool`
- Calculates file size in megabytes and asserts it does not exceed `max_file_size_mb` (100MB).

### `validate_files_count() -> bool`
- Verifies that the temporary `data/uploads/` directory does not exceed `max_files_count` concurrent files.

### `get_file_extension(filename: str) -> str`
- Extracts lowercased file extension (`Path(filename).suffix.lower()`).

### `save_uploaded_file(file_bytes: bytes, filename: str) -> str`
- Validates extension and size.
- Generates a safe collision-free filename: `{stem}_{uuid4[:8]}{suffix}`.
- Writes bytes to disk inside `data/uploads/` and returns the absolute path.

### `delete_file_after_processing(file_path: str)`
- Deletes a specific processed file after ingestion completes.

### `clean_uploads()`
- Removes and recreates the entire `data/uploads/` directory during full memory wipes.

### `read_config(config_path="config/config.yaml") -> dict`
- Helper for loading YAML configurations.

---

## 3. Connections & Component Mapping

- **Imported By:**
  - `src.pipelines.ingestion_pipeline.IngestionPipeline`: Validates, saves, and deletes uploaded files.
  - `src.routers.upload`: Validates files before enqueuing background tasks.
  - `src.routers.sessions`: Cleans uploads folder when memory is cleared.

---

## 4. AI & Developer Guidelines
- Always pair `save_uploaded_file()` with `delete_file_after_processing()` in a `try...finally` block so failed ingestions do not leave residual files on disk.
