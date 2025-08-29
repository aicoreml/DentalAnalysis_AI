# Dental X-Ray Analysis System

An AI-powered system for analyzing dental X-rays using computer vision and natural language processing to generate comprehensive diagnostic reports.

## 🦷 Features

- **AI-Powered Analysis**: Uses state-of-the-art vision models to analyze dental X-rays
- **Comprehensive Reports**: Generates detailed diagnostic reports with findings and recommendations
- **User-Friendly Interface**: Gradio-based web interface for easy interaction
- **Dental Knowledge Integration**: Specialized knowledge base for dental conditions
- **Image Enhancement**: Preprocessing techniques optimized for dental X-rays

## 📁 Project Structure

```
dental-xray-analysis/
├── src/
│   └── app.py              # Main application
├── docs/                   # Documentation
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
├── LICENSE                 # License information
└── README.md               # This file
```

## 🚀 Technology Stack

1. **Vision Encoder**: [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) model for image analysis
2. **LLM Component**: [DialoGPT](https://huggingface.co/microsoft/DialoGPT-medium) for report generation
3. **Image Processing**: OpenCV and PIL for image enhancement
4. **UI Framework**: Gradio for web interface
5. **Backend**: Python with PyTorch and Transformers

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/dental-xray-analysis.git
   cd dental-xray-analysis
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. For GPU support (optional but recommended):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

## ▶️ Usage

Run the application:
```bash
python src/app.py
```

The application will start a web interface where you can upload dental X-rays for analysis.

## 📋 How It Works

1. **Image Analysis**: AI vision model examines the X-ray and identifies dental structures and anomalies
2. **Feature Enhancement**: Applies dental-specific image processing to enhance contrast and details
3. **Diagnostic Generation**: Creates a comprehensive report with differential diagnoses and treatment recommendations
4. **Clinical Guidance**: Provides follow-up protocols and prognostic assessments

## ⚠️ Disclaimer

This is an AI-powered preliminary analysis system. All findings require clinical correlation and professional interpretation for final diagnosis and treatment planning. This system is not intended to replace professional dental diagnosis.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.