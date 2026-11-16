import sys
print(f"Python version: {sys.version}")
try:
    import pandas
    print(f"Pandas: {pandas.__version__}")
except:
    print("Pandas: NOT INSTALLED")
try:
    import sklearn
    print(f"Scikit-learn: {sklearn.__version__}")
except:
    print("Scikit-learn: NOT INSTALLED")
try:
    import streamlit
    print(f"Streamlit: {streamlit.__version__}")
except:
    print("Streamlit: NOT INSTALLED")