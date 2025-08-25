# Dream Composer - Development Setup

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/kalviumcommunity/Dream-Composer.git
cd Dream-Composer
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Project in Development Mode
```bash
pip install -e .
```

### 5. Run Tests
```bash
pytest tokenization/tests/ -v
```

## Project Structure
```
Dream-Composer/
├── tokenization/           # Main tokenization module
│   ├── __init__.py
│   ├── tokenizer.py       # SimpleTokenizer implementation
│   ├── utils.py           # Utility functions
│   └── tests/             # Test files
│       ├── __init__.py
│       └── test_tokenizer.py
├── setup.py               # Package configuration
├── requirements.txt       # Dependencies
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation
```

## Development Notes

- The virtual environment (`venv/`) is excluded from version control for security reasons
- Use `requirements.txt` to manage dependencies
- Run tests before committing changes
- The project is installed in editable mode (`-e`) for development

## Security

This project follows security best practices:
- Virtual environments are not committed to version control
- Dependencies are managed through `requirements.txt`
- Proper `.gitignore` excludes sensitive files and build artifacts
