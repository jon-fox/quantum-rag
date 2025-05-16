# In your terminal (temporarily)
export PYTHONPATH=/mnt/c/Developer_Workspace/quantum_work:$PYTHONPATH

# For permanent setting in .bashrc or .zshrc
echo 'export PYTHONPATH=/mnt/c/Developer_Workspace/quantum_work:$PYTHONPATH' >> ~/.bashrc

# start and navigate to swagger
python app/main.py

http://127.0.0.1:8000/docs#