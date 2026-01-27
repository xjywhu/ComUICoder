import os
import sys
import glob

import re

# Ensure the current directory is in the path to import feedback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feedback import feedback_loop_for_page
try:
    from main import render_single_vue_file, generate_app_file
    from ProcessTools import parse_vue_from_txt_component
except ImportError:
    # Fallback/Mock if running in environment where dependencies aren't perfect
    print("Warning: Could not import main/ProcessTools for final page rendering.")
    render_single_vue_file = None

def update_txt_content_from_vue(page_dir, vue_dir):
    """
    Update the ```component ... ``` block in .txt files in page_dir
    using the content from corresponding .vue files in vue_dir/components.
    Match is done via 'name: "..."' in the txt file.
    """
    print(f"Updating .txt files in {page_dir} from Vue components...")
    txt_files = glob.glob(os.path.join(page_dir, "*.txt"))
    components_dir = os.path.join(vue_dir, "components")
    
    if not txt_files:
        print("No .txt files found to update.")
        return

    count = 0
    for txt_path in txt_files:
        if txt_path.endswith("_errors.txt"):
            continue
            
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find component name in .txt
        # Look for name: "Name" in common patterns
        match = re.search(r'name:\s*["\']([^"\']+)["\']', content)
        
        if match:
            comp_name = match.group(1)
            vue_path = os.path.join(components_dir, f"{comp_name}.vue")
            
            if os.path.exists(vue_path):
                with open(vue_path, 'r', encoding='utf-8') as vf:
                    vue_code = vf.read()
                
                # Replace the component block
                # We assume the format is ```component\n<code>\n```
                new_block = f"```component\n{vue_code}\n```"
                
                # Regex replace
                content_new = re.sub(r'```component\s*.*?\s*```', lambda x: new_block, content, count=1, flags=re.DOTALL)
                
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(content_new)
                count += 1
                print(f"Updated {os.path.basename(txt_path)} from {comp_name}.vue")
            else:
                print(f"Warning: Corresponding Vue file not found for {comp_name} ({vue_path})")
        else:
            print(f"Skipping {os.path.basename(txt_path)} - could not find component name")
            
    print(f"Updated {count} files.")


def example_3_run_page_feedback_loop():
    """
    Example 3: Running the feedback loop for an entire page of components.
    This scans a directory for multiple components, evaluates them all, 
    and then attempts to fix the 'top_k' worst-performing ones.
    
    Scenario: You have generated a whole page with many sub-components and want to batch-improve the bad ones.
    """
    print("\nRunning Example 3: Page Feedback Loop")

    # == Configuration (Replace with actual paths) ==
    
    # User provided paths
    # Page Dir: C:\Users\Shuoqi\Documents\GitHub\VueGen\gen+gt\1\1_cropped (inferred)
    # The user gave ...\gen+gt\1, and based on ls, 1_cropped is inside it.
    gen_root = r"C:\Users\Shuoqi\Documents\GitHub\VueGen\gen+gt"
    page_idx = "1"
    page_dir = os.path.join(gen_root, page_idx, f"{page_idx}_cropped")
    tmp_dir = os.path.join(gen_root, "tmp")
    
    # Vue Project and Driver
    base_dir = r"C:\Users\Shuoqi\Documents\GitHub\VueGen\VueGen" # Codebase root
    vue_dir = os.path.join(base_dir, "my-vue-app", "src")
    proj_dir = os.path.join(base_dir, "my-vue-app")
    webdriver_path = r"C:\Program Files\geckodriver.exe"
    
    print(f"Target Page Directory: {page_dir}")
    
    # Sync .txt files with latest Vue code
    # This ensures that the feedback loop evaluates the current state of the Vue content
    update_txt_content_from_vue(page_dir, vue_dir)
    
    try:
        # This will evaluate all valid components in page_dir and fix the worst 3
        feedback_loop_for_page(
            page_dir=page_dir,
            vue_dir=vue_dir,
            proj_dir=proj_dir,
            webdriver_path=webdriver_path,
            max_iterations=2,
            top_k=3
        )
        print("Page feedback loop completed.")

        # --- Generate Final Full Page Screenshot ---
        if render_single_vue_file:
            print("\nGenerating final full page screenshot...")
            
            # 1. Parse all latest component .txt files in page_dir
            # (Note: feedback loop updates the .txt files in place or creates _best.txt)
            # We assume we want to view the current state of page_dir
            txt_files = glob.glob(os.path.join(page_dir, "*.txt"))
            txt_files = [f for f in txt_files if not f.endswith('_errors.txt')] # Exclude error logs
            
            if txt_files:
                components_dir = os.path.join(vue_dir, "components")
                os.makedirs(components_dir, exist_ok=True)
                
                # Extract Vue code from all components
                for txt_p in txt_files:
                    try:
                        parse_vue_from_txt_component(txt_p, vue_dir, "component")
                    except Exception as parse_err:
                        print(f"Warning: Failed to parse {os.path.basename(txt_p)}: {parse_err}")

                # 2. Re-generate App.vue to include ALL components
                # Find all .vue files in the components directory
                all_vue_files = [f for f in os.listdir(components_dir) if f.endswith('.vue') and f != 'App.vue']
                
                if all_vue_files:
                    tmp_app = generate_app_file(components_dir, all_vue_files)
                    if os.path.exists(tmp_app):
                        final_app_path = os.path.join(components_dir, "App.vue")
                        os.replace(tmp_app, final_app_path)
                        
                        # 3. Render the full page
                        final_screenshot = os.path.join(page_dir, "full_page_final.png")
                        final_html = os.path.join(page_dir, "full_page_final.html")
                        
                        render_single_vue_file(proj_dir, components_dir, final_html, final_screenshot, webdriver_path)
                        print(f"Final full page screenshot saved to: {final_screenshot}")
                else:
                    print("No Vue components found to render.")
            else:
                print("No component .txt files found in page_dir.")
                
    except Exception as e:
        print(f"Error running page feedback loop: {e}")

if __name__ == "__main__":
    example_3_run_page_feedback_loop()
