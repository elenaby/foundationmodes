# # import gigapath
# # slide_encoder = gigapath.slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536)
# # print("Success! Slide encoder created.")

# # gigapath_encoder_discovery.py
# # discover_gigapath_api.py
import gigapath.slide_encoder
import inspect

print("=== Discovering GigaPath Slide Encoder API ===")
print(f"Module: {gigapath.slide_encoder.__file__}")

# 1. List all public attributes
print("\n1. All public attributes in slide_encoder:")
for attr_name in dir(gigapath.slide_encoder):
    if not attr_name.startswith('_'):
        attr = getattr(gigapath.slide_encoder, attr_name)
        print(f"   {attr_name}: {type(attr).__name__}")

# 2. Check SlideEncoder class if it exists
print("\n2. SlideEncoder class analysis:")
if hasattr(gigapath.slide_encoder, 'SlideEncoder'):
    SlideEncoder = gigapath.slide_encoder.SlideEncoder
    print(f"   Found SlideEncoder class")
    
    # Check constructor
    print(f"   Constructor signature: {inspect.signature(SlideEncoder.__init__)}")
    
    # Check class methods
    print(f"   Class methods:")
    for attr_name in dir(SlideEncoder):
        if not attr_name.startswith('_'):
            attr = getattr(SlideEncoder, attr_name)
            if callable(attr):
                try:
                    sig = inspect.signature(attr)
                    print(f"     - {attr_name}{sig}")
                except:
                    print(f"     - {attr_name}()")

# 3. Look for model creation functions
print("\n3. Model creation functions:")
functions = []
for attr_name in dir(gigapath.slide_encoder):
    if not attr_name.startswith('_'):
        attr = getattr(gigapath.slide_encoder, attr_name)
        if callable(attr) and not isinstance(attr, type):  # Function, not class
            func_name = attr_name.lower()
            if any(keyword in func_name for keyword in ['create', 'build', 'make', 'from', 'load']):
                functions.append(attr_name)

if functions:
    for func_name in functions:
        func = getattr(gigapath.slide_encoder, func_name)
        try:
            sig = inspect.signature(func)
            print(f"   ✅ {func_name}{sig}")
        except:
            print(f"   ✅ {func_name}()")
else:
    print("   No obvious model creation functions found")

# 4. Search for configuration classes
print("\n4. Configuration classes:")
config_classes = []
for attr_name in dir(gigapath.slide_encoder):
    if not attr_name.startswith('_'):
        attr = getattr(gigapath.slide_encoder, attr_name)
        if isinstance(attr, type) and 'config' in attr_name.lower():
            config_classes.append(attr_name)

for cls_name in config_classes:
    cls = getattr(gigapath.slide_encoder, cls_name)
    print(f"   {cls_name}: {cls}")

# 5. Try to find usage in the codebase
print("\n5. Checking for example usage...")
import os
example_files = []
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".py") and any(name in file.lower() for name in ['example', 'test', 'demo']):
            example_files.append(os.path.join(root, file))

if example_files:
    print(f"   Found example files: {example_files[:3]}")
    for file in example_files[:2]:  # Check first 2
        try:
            with open(file, 'r') as f:
                content = f.read()
                if 'slide_encoder' in content:
                    print(f"   Checking {file}...")
                    # Extract relevant lines
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'slide_encoder' in line and 'import' not in line:
                            print(f"     Line {i+1}: {line.strip()}")
        except:
            pass

#LoRA
import gigapath

slide_encoder = gigapath.slide_encoder.create_model(
    "hf_hub:prov-gigapath/prov-gigapath",
    "gigapath_slide_enc12l768d",
    1536
)
print(type(slide_encoder))
print(slide_encoder)