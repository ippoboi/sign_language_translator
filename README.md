# Sign Language Translator

A Python-based application that translates sign language gestures into text using computer vision and machine learning.

## Features

- Real-time sign language detection
- Translation of hand gestures to text
- Support for [specify which sign language system, e.g., ASL]
- Built with MediaPipe for hand tracking
- Powered by PyTorch for gesture recognition

## Prerequisites

- Python 3.11+
- Webcam for gesture input

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ippoboi/sign_language_translator.git
cd sign_language_translator
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Activate the virtual environment (if not already activated):

```bash
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Run the application:

```bash
cd sign_language_test
python backend.py
```

3. Position your hand in front of the webcam and make gestures.
