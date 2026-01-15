import os
import re

posts_dir = './2025'

def fix_content_with_report(file_path, content):
    # Regex for Block math, Inline math, and LaTeX environments
    math_patterns = [
        r'(\$\$.*?\$\$)',
        r'((?<!\\)\$.*?(?<!\\)\$)',
        r'(\\begin\{.*?\}.*?\\end\{.*?\})'
    ]
    combined_pattern = '|'.join(math_patterns)
    
    replacements = []

    def replace_in_math(match):
        found_math = match.group(0)
        if '{{' in found_math or '}}' in found_math:
            fixed = found_math.replace('{{', '{ { ').replace('}}', ' } }')
            # Store a snippet for the report
            replacements.append((found_math.strip().split('\n')[0][:60], 
                                 fixed.strip().split('\n')[0][:60]))
            return fixed
        return found_math

    new_content = re.compile(combined_pattern, re.DOTALL).sub(replace_in_math, content)
    return new_content, replacements

def run_with_visual_report(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    old_text = f.read()
                
                new_text, changes = fix_content_with_report(path, old_text)
                
                if changes:
                    print(f"\n📄 FILE: {file}")
                    print("-" * 40)
                    for original, fixed in changes:
                        print(f"  [OLD]: {original}...")
                        print(f"  [NEW]: {fixed}...")
                    
                    # Write the changes to the file
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_text)

if __name__ == "__main__":
    run_with_visual_report(posts_dir)
