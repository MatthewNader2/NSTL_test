import sys

with open("src/inference.py", "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if 'def generate_text(self, prompt: str, max_tokens: int = 2048, schema: dict = None) -> str:' in line:
        new_lines.append('    def generate_text(self, prompt: str, max_tokens: int = 2048, schema: dict = None, system_prompt: str = None) -> str:\n')
    elif 'messages = [{"role": "user", "content": prompt}]' in line:
        new_lines.append('        messages = []\n')
        new_lines.append('        if system_prompt:\n')
        new_lines.append('            messages.append({"role": "system", "content": system_prompt})\n')
        new_lines.append('        messages.append({"role": "user", "content": prompt})\n')
    elif 'def feedback_check(self, generated_code: str) -> str:' in line:
        new_lines.append(line)
        # We need to replace the body of feedback_check.
        # Find the end of feedback_check
    else:
        new_lines.append(line)

with open("src/inference.py", "w") as f:
    f.writelines(new_lines)
