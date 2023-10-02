import sys
import tensorflow as tf 

def break_clause():
    ans = input("proceed(y/n)> ")
    if ans == "y":
        print('Approved')
    else: sys.exit(0) 

def save_model():
    print("Do you want to save the model?")
    break_clause()
    model = input("modle name > ")
    name = input("save as: ")
    model.save(name)