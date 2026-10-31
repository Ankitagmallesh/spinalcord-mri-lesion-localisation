import glob
for f in glob.glob('*.py'):
    if f == 'fix.py' or f == 'generate_synthetic_data.py':
        continue
    with open(f, 'r') as file:
        content = file.read()
    if content.startswith('"') and content.endswith('"'):
        content = content[1:-1]
        with open(f, 'w') as file:
            file.write(content)
