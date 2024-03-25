from util.transform import *

def main():
    f = open("./data/cryptoVocab.txt", "r")
    crypto_vocabulary = f.read().split(',')
    crypto_vocabulary = [term.replace('"', '') for term in crypto_vocabulary]
    
     

if __name__ == "__main__":
    main()