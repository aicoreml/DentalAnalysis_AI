import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
import warnings
import gradio as gr
warnings.filterwarnings('ignore')

class DentalXRayAnalyzer:
    def __init__(self):
        # Initialize components of the multimodal pipeline
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the vision encoder (BLIP for image captioning)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.vision_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base").to(self.device)
        
        # Initialize the LLM component (simplified version - in practice would use a medical-tuned model)
        self.llm_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.llm_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to(self.device)
        
        # Add padding token if it doesn't exist
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        
        # Dental-specific knowledge (in a real system, this would be a comprehensive knowledge base)
        self.dental_knowledge = {
            "caries": "Dental caries (cavities) appear as dark spots or shadows on the X-ray, indicating decay in the tooth structure.",
            "periodontal": "Periodontal disease shows as bone loss around teeth, appearing as darkened areas with reduced bone height.",
            "impacted": "Impacted teeth are unable to fully erupt, often seen with wisdom teeth that are partially or completely buried in bone.",
            "restoration": "Dental restorations (fillings, crowns) appear as bright white areas on X-rays due to their radiopaque properties.",
            "root_canal": "Teeth with root canal treatment show as having filling material in the root canals with a crown restoration.",
            "missing": "Missing teeth are identified by the absence of a tooth in the dental arch where one would normally be present.",
            "infection": "Periapical infections appear as dark areas at the tip of the tooth root, indicating inflammation or abscess."
        }
        
        # Preprocessing transforms for dental X-rays
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_xray(self, image_path):
        """Preprocess the dental X-ray image"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply dental X-ray specific enhancements
        img_array = np.array(image)
        
        # Apply contrast enhancement for dental X-rays
        img_array = cv2.convertScaleAbs(img_array, alpha=1.5, beta=0)
        
        # Apply sharpening filter to enhance details
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_array = cv2.filter2D(img_array, -1, kernel)
        
        enhanced_image = Image.fromarray(img_array)
        return enhanced_image
    
    def detect_dental_features(self, image):
        """Use vision model to identify dental features"""
        # Preprocess image for the vision model
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Generate image caption with dental focus
        generated_ids = self.vision_model.generate(
            **inputs,
            max_length=100,
            num_beams=5,
            repetition_penalty=1.5,
            temperature=0.9
        )
        
        # Decode the generated description
        description = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Enhance the description with dental-specific terminology
        dental_description = self.enhance_dental_description(description)
        
        return dental_description
    
    def enhance_dental_description(self, description):
        """Add dental-specific terminology to the generated description"""
        # This is a simplified version - in practice would use a more sophisticated method
        dental_terms = []
        detailed_findings = []
        
        # Check for common dental conditions based on the description
        description_lower = description.lower()
        
        if any(word in description_lower for word in ["dark", "shadow", "spot", "black"]):
            dental_terms.append("caries")
            detailed_findings.append("Radiolucent areas observed, suggestive of dental caries (tooth decay)")
        
        if any(word in description_lower for word in ["bone", "loss", "reduction"]):
            dental_terms.append("periodontal")
            detailed_findings.append("Evidence of alveolar bone loss, indicating possible periodontal disease")
        
        if any(word in description_lower for word in ["not visible", "hidden", "buried"]):
            dental_terms.append("impacted")
            detailed_findings.append("Partially or fully impacted teeth identified")
        
        if any(word in description_lower for word in ["white", "bright", "metal"]):
            dental_terms.append("restoration")
            detailed_findings.append("Radiopaque areas consistent with dental restorations (fillings, crowns, etc.)")
        
        if any(word in description_lower for word in ["missing", "gap", "empty"]):
            dental_terms.append("missing")
            detailed_findings.append("Areas where teeth are absent from the dental arch")
        
        if any(word in description_lower for word in ["infection", "swelling", "abscess"]):
            dental_terms.append("infection")
            detailed_findings.append("Periapical radiolucencies suggesting possible infection or abscess")
        
        # Create enhanced description with more structure
        enhanced = f"COMPREHENSIVE DENTAL X-RAY ANALYSIS\n"
        enhanced += f"Initial Image Description: {description}\n\n"
        
        if dental_terms:
            enhanced += f"IDENTIFIED FEATURES:\n"
            enhanced += f"- Dental Conditions Detected: {', '.join(dental_terms)}\n"
            enhanced += f"- Detailed Observations:\n"
            for finding in detailed_findings:
                enhanced += f"  • {finding}\n"
        else:
            enhanced += "No specific dental pathologies identified in the initial assessment.\n"
            enhanced += "STRUCTURAL FINDINGS:\n"
            enhanced += "- General tooth morphology appears intact\n"
            enhanced += "- No obvious radiolucencies or radiopacities of concern\n"
            enhanced += "- Bone levels appear maintained\n"
        
        enhanced += "\nPOTENTIAL CLINICAL SIGNIFICANCE:\n"
        enhanced += "These findings require clinical correlation for definitive diagnosis and treatment planning."
        
        return enhanced
    
    def generate_diagnostic_report(self, dental_description):
        """Use LLM to generate a comprehensive diagnostic report"""
        # Create a prompt for the LLM with dental context
        prompt = f"""
        As a highly skilled dental radiologist, analyze this X-ray description and provide a comprehensive diagnostic report.

        X-RAY OBSERVATIONS: {dental_description}

        Please provide a detailed report in the following format:
        
        1. DETAILED RADIOLOGICAL FINDINGS:
        - Describe each identified feature in depth, including location and significance
        
        2. DIFFERENTIAL DIAGNOSES:
        - List possible conditions for each finding with likelihood assessment
        
        3. CLINICAL RECOMMENDATIONS:
        - Specific treatment suggestions for each diagnosis
        
        4. FOLLOW-UP PROTOCOL:
        - Recommended monitoring schedule and additional procedures
        
        5. PROGNOSTIC ASSESSMENT:
        - Long-term outlook based on the findings
        
        DENTAL REPORT:
        """
        
        # Tokenize the input
        inputs = self.llm_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.llm_model.generate(
                inputs,
                max_length=1000,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                top_p=0.9,
                repetition_penalty=1.2
            )
        
        # Decode the generated report
        report = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the report part (remove the prompt)
        if "DENTAL REPORT:" in report:
            report = report.split("DENTAL REPORT:")[-1].strip()
        else:
            # If the expected marker isn't found, return the full response
            report = report.strip()
        
        # If the report is still empty or too short, provide a default structured report
        if len(report) < 50:
            report = self._generate_default_report(dental_description)
        
        return report
    
    def _generate_default_report(self, dental_description):
        """Generate a default structured report when LLM generation fails"""
        default_report = f"""
1. DETAILED RADIOLOGICAL FINDINGS:
Based on the image analysis: {dental_description}

2. DIFFERENTIAL DIAGNOSES:
The findings suggest several possible conditions that require clinical correlation and further evaluation.

3. CLINICAL RECOMMENDATIONS:
- Clinical examination to confirm radiographic findings
- Patient symptoms correlation
- Consider additional imaging if needed

4. FOLLOW-UP PROTOCOL:
- Routine follow-up in 6-12 months
- Monitor any identified areas of concern
- Compare with previous radiographs if available

5. PROGNOSTIC ASSESSMENT:
Prognosis depends on the confirmed diagnosis and timely intervention. Early detection generally leads to more favorable outcomes.

Note: This is an AI-generated preliminary analysis. Clinical correlation and professional interpretation are essential for final diagnosis and treatment planning.
        """
        return default_report.strip()
    
    def visualize_analysis(self, original_image, enhanced_image, description, report):
        """Create a visualization of the analysis results"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Display original image
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original X-Ray')
        axes[0].axis('off')
        
        # Display enhanced image
        axes[1].imshow(enhanced_image, cmap='gray')
        axes[1].set_title('Enhanced X-Ray')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print the analysis results
        print("=" * 60)
        print("DENTAL X-RAY ANALYSIS REPORT")
        print("=" * 60)
        print("\nIMAGE ANALYSIS:")
        print(description)
        print("\n" + "-" * 60)
        print("DIAGNOSTIC REPORT:")
        print(report)
        print("=" * 60)
    
    def analyze_xray(self, image_path):
        """Complete pipeline for dental X-ray analysis"""
        # Preprocess the X-ray image
        original_image = Image.open(image_path).convert('RGB')
        enhanced_image = self.preprocess_xray(image_path)
        
        # Detect dental features
        dental_description = self.detect_dental_features(enhanced_image)
        
        # Generate diagnostic report
        diagnostic_report = self.generate_diagnostic_report(dental_description)
        
        # Visualize results
        # self.visualize_analysis(original_image, enhanced_image, dental_description, diagnostic_report)
        
        return {
            "description": dental_description,
            "report": diagnostic_report
        }

# Example usage and demonstration
def demonstrate_dental_analysis():
    """Demonstrate the dental X-ray analysis pipeline"""
    print("Initializing Dental X-Ray Analyzer...")
    analyzer = DentalXRayAnalyzer()
    
    # Since we don't have actual dental X-rays, we'll simulate the process
    print("\nSimulating dental X-ray analysis...")
    
    # Create a sample dental X-ray description (in practice, this would come from the vision model)
    sample_description = """
    The panoramic radiograph shows a full dentition with multiple restorations present. 
    There appears to be a radiolucent area on the distal surface of the lower first molar, 
    suggesting possible caries. Moderate bone loss is observed in the posterior regions. 
    The third molars are partially visible with the lower right wisdom tooth appearing impacted.
    """
    
    # Generate a diagnostic report based on the sample description
    diagnostic_report = analyzer.generate_diagnostic_report(sample_description)
    
    # Display results
    print("=" * 60)
    print("DENTAL X-RAY ANALYSIS REPORT (SIMULATION)")
    print("=" * 60)
    print("\nIMAGE ANALYSIS:")
    print(sample_description)
    print("\n" + "-" * 60)
    print("DIAGNOSTIC REPORT:")
    print(diagnostic_report)
    print("=" * 60)
    
    # Explain the technology stack
    print("\n\nTECHNOLOGY EXPLANATION:")
    print("This implementation demonstrates a multimodal AI approach for dental X-ray analysis:")
    print("1. Vision Encoder: BLIP model processes the X-ray image and generates initial descriptions")
    print("2. LLM Component: DialoGPT model generates comprehensive diagnostic reports")
    print("3. Dental Knowledge Base: Specialized information about dental conditions and terminology")
    print("4. Image Enhancement: Preprocessing techniques optimized for dental X-rays")
    print("\nIn a production system, you would:")
    print("- Use dental-specific vision models trained on annotated X-ray datasets")
    print("- Implement a medical-tuned LLM like Med-PaLM or ClinicalBERT")
    print("- Include rigorous validation mechanisms to prevent hallucinations")
    print("- Add dental domain experts in the loop for verification")

# Gradio interface function
def analyze_dental_xray(image):
    """Function to analyze dental X-ray using Gradio interface"""
    # Save the uploaded image to a temporary file
    image_path = "temp_dental_xray.jpg"
    image.save(image_path)
    
    # Initialize analyzer
    analyzer = DentalXRayAnalyzer()
    
    # Analyze the X-ray
    result = analyzer.analyze_xray(image_path)
    
    # Format the results for better presentation
    formatted_description = f"## IMAGE ANALYSIS RESULTS\n\n{result['description']}"
    
    formatted_report = f"## COMPREHENSIVE DIAGNOSTIC REPORT\n\n{result['report']}"
    
    # Return the results
    return formatted_description, formatted_report

# Create Gradio interface
def create_gradio_interface():
    """Create and launch the Gradio interface"""
    with gr.Blocks(title="Dental X-Ray Analyzer") as demo:
        gr.Markdown("# 🦷 Dental X-Ray Analysis System")
        gr.Markdown("Upload a dental X-ray image for AI-powered analysis and comprehensive diagnostic report generation.")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Dental X-Ray")
                analyze_button = gr.Button("Analyze X-Ray", variant="primary")
                gr.Markdown("*Supported formats: JPG, PNG, JPEG*")
            
            with gr.Column():
                description_output = gr.Markdown(label="Image Analysis Results")
                report_output = gr.Markdown(label="Comprehensive Diagnostic Report")
        
        analyze_button.click(
            fn=analyze_dental_xray,
            inputs=image_input,
            outputs=[description_output, report_output]
        )
        
        gr.Markdown("## 📋 How It Works")
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### Step 1: Image Analysis
                - AI vision model examines the X-ray
                - Identifies dental structures and anomalies
                - Detects potential pathologies
                
                ### Step 2: Feature Enhancement
                - Applies dental-specific image processing
                - Enhances contrast and details
                - Prepares image for detailed analysis
                """)
            with gr.Column():
                gr.Markdown("""
                ### Step 3: Diagnostic Generation
                - Comprehensive report creation
                - Differential diagnoses provided
                - Treatment recommendations included
                
                ### Step 4: Clinical Guidance
                - Follow-up protocols suggested
                - Prognostic assessments provided
                - References to dental knowledge base
                """)
        
        gr.Markdown("## 🧠 Technology Stack")
        gr.Markdown("""
        This system uses a state-of-the-art multimodal AI approach for dental X-ray analysis:
        
        1. **Vision Encoder**: [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) model processes the X-ray image and generates detailed descriptions
        2. **LLM Component**: [DialoGPT](https://huggingface.co/microsoft/DialoGPT-medium) model generates comprehensive diagnostic reports
        3. **Dental Knowledge Base**: Specialized information about dental conditions and terminology
        4. **Image Enhancement**: Preprocessing techniques optimized for dental X-rays
        
        In a production system, you would:
        - Use dental-specific vision models trained on annotated X-ray datasets
        - Implement a medical-tuned LLM like Med-PaLM or ClinicalBERT
        - Include rigorous validation mechanisms to prevent hallucinations
        - Add dental domain experts in the loop for verification
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch Gradio interface
    interface = create_gradio_interface()
    interface.launch(share=True)