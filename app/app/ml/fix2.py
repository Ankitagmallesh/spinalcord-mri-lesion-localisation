import glob
for f in glob.glob('*.py'):
    if f in ['fix.py', 'fix2.py', 'generate_synthetic_data.py']:
        continue
    with open(f, 'r') as file:
        content = file.read()
    
    content = content.replace('\\"', '"')
    
    with open(f, 'w') as file:
        file.write(content)
