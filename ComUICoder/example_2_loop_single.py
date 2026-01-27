import os
import sys

# Ensure the current directory is in the path to import feedback
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feedback import feedback_loop_for_component

def example_2_run_feedback_loop():
    """
    Example 2: Running the full feedback loop.
    This performs the cycle: 
    Render -> Screenshot -> Evaluate -> Generate Fix Prompt -> LLM Fix -> Render...
    
    Scenario: You want to automatically improve a generated component.
    """
    print("\nRunning Example 2: Full Feedback Loop")

    # == Configuration (Replace with actual paths) ==
    txt_path = "output/generated_code.txt"  # The text file containing the LLM's initial Vue code
    vue_dir = "vue_project_template/src"    # Where to inject the code for rendering
    gt_image_path = "dataset/images/gt.png" # The target image
    proj_dir = "vue_project_template"       # The root of the Vue project used for rendering
    webdriver_path = r"C:\Program Files\geckodriver.exe" 
    
    # Run the loop
    # This function handles the logic of calling the renderer and the LLM fixer
    try:
        feedback_loop_for_component(
            txt_path=txt_path,
            vue_dir=vue_dir,
            gt_image_path=gt_image_path,
            proj_dir=proj_dir,
            webdriver_path=webdriver_path,
            max_iterations=2, # How many rounds of fixing to attempt
            model="gemini-2.5-pro"
        )
        print("Feedback loop completed.")
        
        # The best result screenshot is automatically saved by feedback_loop_for_component
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        crop_dir = os.path.dirname(txt_path)
        best_screenshot = os.path.join(crop_dir, f"{base_name}_best_generated.png")
        if os.path.exists(best_screenshot):
            print(f"Final best screenshot saved to: {best_screenshot}")
        
    except Exception as e:
        print(f"Error running feedback loop (check paths and dependencies): {e}")

if __name__ == "__main__":
    example_2_run_feedback_loop()
