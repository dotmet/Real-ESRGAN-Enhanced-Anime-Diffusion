import os

def check_RealESRGAN():
    try:
        from RealESRGANv030.interface import realEsrgan
    except:
        os.system('cd RealESRGANv030 && python setup.py develop')
        
if __name__ == '__main__':
    check_RealESRGAN()
    status = os.system('python app.py')
    if status==256:
        print('Run failed, try set MKL_THREADING_LAYER=GNU\n')
        os.system('export MKL_THREADING_LAYER=GNU && python app.py')
