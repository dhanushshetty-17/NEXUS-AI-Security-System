import os
import json
import time
import threading
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    _GENAI_AVAILABLE = False

class GenAIReporter:
    """
    Generates police-style incident reports using Google Gemini Vision AI.
    Runs asynchronously to avoid blocking the main video pipeline.
    """
    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_key = os.getenv("GEMINI_API_KEY")
        if _GENAI_AVAILABLE and self.api_key and self.api_key != "your_google_gemini_api_key_here":
            try:
                self.client = genai.Client(api_key=self.api_key)
                self.enabled = True
            except Exception:
                self.client = None
                self.enabled = False
        else:
            self.client = None
            self.enabled = False

    def generate_report_async(self, image_path: str, context: dict) -> None:
        """Dispatches the report generation to a background thread."""
        if not self.enabled:
            print("[GenAI] Skipping report generation: GEMINI_API_KEY not set.")
            return

        thread = threading.Thread(
            target=self._generate_and_save_report,
            args=(image_path, context),
            daemon=True
        )
        thread.start()

    def _generate_and_save_report(self, image_path: str, context: dict) -> None:
        if not os.path.exists(image_path):
            print(f"[GenAI] Image not found: {image_path}")
            return

        try:
            # Upload the file to Gemini
            sample_file = self.client.files.upload(file=image_path)
            
            prompt = (
                "You are an expert AI police dispatcher and security analyst. "
                f"A threat was detected: {context.get('label', 'Unknown')} "
                f"with severity {context.get('severity', 'Unknown')}. "
                "Look at the provided security camera snapshot and write a short, professional "
                "incident report. Include a visual description of the suspect/scene, the nature of the threat, "
                "and recommended immediate actions. Do not use markdown formatting, just plain text."
            )
            
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[sample_file, prompt]
            )
            
            report = {
                "timestamp": time.time(),
                "camera_id": context.get("camera_id", "unknown"),
                "threat_label": context.get("label", "Unknown"),
                "severity": context.get("severity", "Unknown"),
                "image_path": image_path,
                "ai_summary": response.text.strip(),
            }
            
            # Save report
            report_filename = f"report_{int(time.time())}.json"
            report_path = self.output_dir / report_filename
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
                
            print(f"[GenAI] Saved incident report to {report_path}")
            
        except Exception as e:
            print(f"[GenAI] Failed to generate report: {e}")
