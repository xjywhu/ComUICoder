import os
import sys

# Ensure the current directory is in the path to import feedback
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feedback import (
    evaluate_component_with_uied, 
    generate_error_feedback_text,
    annotate_image_with_errors
)

def example_1_evaluate_single_component():
    """
    Example 1: Basic evaluation of a generated component against a Ground Truth (GT) image.
    This calculates the similarity score and finds specific inconsistencies using UIED detection.
    
    Scenario: You have a GT image and a Generated image, and you want to know how well they match
    and where the errors are.
    """
    print("Running Example 1: Single Component Evaluation")
    
    # == Configuration (Replace with actual paths) ==
    gt_image_path = "dataset/images/ground_truth.png"       # Path to the real design image
    generated_image_path = "output/generated_screenshot.png" # Path to the screenshot of generated code
    vue_code_path = "output/component.vue"                  # Path to the generated Vue code (optional)
    output_dir = "feedback_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Evaluate: Detect widgets and compare
    # This involves heavy lifting: running UIED (Unsupervised component detection) on both images
    print("Evaluating...")
    result = evaluate_component_with_uied(
        gt_image_path=gt_image_path,
        generated_image_path=generated_image_path,
        vue_code_path=vue_code_path,
        output_dir=output_dir
    )

    if result:
        # result dictionary contains: match_score, gt_widgets, gen_widgets, inconsistencies
        print(f"Match Score: {result['match_score']:.2f} (0.0 to 1.0)")
        print(f"Inconsistencies Found: {len(result['inconsistencies'])}")
        
        # 2. Generate Visual Feedback: Draw red boxes on the GT image where errors are
        annotated_gt_path = os.path.join(output_dir, "annotated_gt.png")
        annotate_image_with_errors(
            image_path=gt_image_path,
            inconsistencies=result['inconsistencies'],
            output_path=annotated_gt_path,
            is_gt=True
        )
        print(f"Annotated GT image saved to: {annotated_gt_path}")

        # 3. Generate Text Feedback: Create a readable report of errors for the LLM
        feedback_txt_path = os.path.join(output_dir, "feedback.txt")
        generate_error_feedback_text(
            inconsistencies=result['inconsistencies'],
            output_path=feedback_txt_path,
            vue_code_path=vue_code_path
        )
        print(f"Text feedback saved to: {feedback_txt_path}")
    else:
        print("Evaluation failed (one or both images could not be loaded).")

if __name__ == "__main__":
    example_1_evaluate_single_component()
