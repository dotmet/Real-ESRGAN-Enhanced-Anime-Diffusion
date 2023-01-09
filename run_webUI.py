import os
import utils
        
if __name__ == '__main__':
    utils.check_RealESRGAN()
    status = os.system('python app.py')
    if status==256:
        print('Run failed, try set MKL_THREADING_LAYER=GNU\n')
        os.system('export MKL_THREADING_LAYER=GNU && python app.py')
