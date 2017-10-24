import numpy as np
from io import BytesIO
from captcha.image import ImageCaptcha
import os
from random import randint

path = 'D:\\Documents\\CS229\\Project\\Fonts\\'
database = 'D:\\Documents\\CS229\\Project\\SingleCharDatabase\\'
fonts = [];
for file in os.listdir(path):
    fonts.append(path+file);
print(fonts)
image = ImageCaptcha(fonts=fonts)

captchlength = 1;
numGenerate = 1000;
generatorDictionary = dict();
for i in range(numGenerate):
    #generate random number and convert to character
    sample = [randint(65,90) for x in range(captchlength)];
    #convert to charac6ters
    string = [chr(x) for x in sample];
    string = ''.join(string);
    print(string)
    if(string not in generatorDictionary):
        generatorDictionary[string] = 0;
    generatorDictionary[string]+=1;
    data = image.generate(string)
    assert isinstance(data, BytesIO)

    ## before we write, we should check for duplicates, which there will be in the
    ## case we use a single char

    key = generatorDictionary[string];
    image.write(string, database+string+'_'+str(key)+'.png')