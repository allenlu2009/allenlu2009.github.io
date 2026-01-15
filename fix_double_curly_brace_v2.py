import os
import re

posts_dir = './2025'

def fix_content_minimalist(content):
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
        if '{{' in found_math:
            # ONLY replace {{ with { {
            fixed = found_math.replace('{{', '{ {')
            
            # Store a snippet for the report
            orig_snip = found_math.strip().split('\n')[0][:50]
            fixed_snip = fixed.strip().split('\n')[0][:50]
            replacements.append((orig_snip, fixed_snip))
            return fixed
        return found_math

    return re.compile(combined_pattern, re.DOTALL).sub(replace_in_math, content), replacements

def run_minimalist_fix(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    old_text = f.read()
                
                new_text, changes = fix_content_minimalist(old_text)
                
                if changes:
                    print(f"\n📄 {file}")
                    for original, fixed in changes:
                        print(f"  [OLD]: {original}")
                        print(f"  [NEW]: {fixed}")
                    
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_text)

if __name__ == "__main__":
    run_minimalist_fix(posts_dir)
