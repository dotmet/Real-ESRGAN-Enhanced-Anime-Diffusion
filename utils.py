def is_google_colab():
    try:
        import google.colab
        return True
    except:
        return False

import os
       
def check_RealESRGAN():
    try:
        from RealESRGANv030.interface import realEsrgan
    except:
        os.system('cd RealESRGANv030 && python setup.py develop')

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n', 
                        type = int, 
                        default = 0,
                        help = 'Generate image n times, (0=infinite)')
    parser.add_argument('-mn', '--model_name', 
                        type = str,
                        default = 'anything v3',
                        help = 'Models used to generate anime images.'
                        )
    parser.add_argument('-wd', '--words',
                        type = str,
                        default = "1girl, brown hair, green eyes, colorful, autumn, \
cumulonimbus clouds, lighting, blue sky, falling leaves, garden",
                        help = 'Text (prompt keys) or the name of a file which contains text(s) used to generate anime image'
                        )
    parser.add_argument('-nwd', '--neg_words',
                        type = str,
                        default = "lowers, bad anatomy, bad hands, text, error, \
missing fingers, extra digit, fewer digits, cropped, worst quality, \
low quality, normal quality, jpeg artifacts, signature, watermark, \
username, blurry, artist name, bad feet",
                        help = 'Negative prompt keys'
                        )
    parser.add_argument('-img', '--image',
                        type = str,
                        default = '',
                        help = 'Input image used to generate anime figure.')
    parser.add_argument('-w', '--width',
                        type = int,
                        default = 512,
                        help = 'Image width before resolution scale',
                        )
    parser.add_argument('-h_', '--height',
                        type = int,
                        default = 512,
                        help = 'Image height before resolution scale')
    parser.add_argument('-g', '--guidance',
                        type = float,
                        default = 7.5,
                        help = 'Guidance for anime image generation.')
    parser.add_argument('-st', '--strength',
                        type = float,
                        default = 0.5,
                        help = 'Transform strength.')
    parser.add_argument('-gs', '--gen_steps',
                        type = int,
                        default = 20,
                        help = 'Steps for anime image generation process.')
    parser.add_argument('-seed',
                        type = int, 
                        default = 0,
                        help = 'Random seed, 0 means random')
    parser.add_argument('-s', '--scale',
                        type = int,
                        default = 4,
                        help = 'Resolution scale (> 1)')
    parser.add_argument('-t', '--tile',
                        type = int,
                        default = 0, 
                        help = 'Tile for resolution up scale, 0 = no tile')
    parser.add_argument('-e', '--extension',
                        type = str,
                        default = 'auto', 
                        help = 'extension of output image')
    parser.add_argument('-o', '--out_dir',
                        type = str,
                        default = 'imgs', 
                        help = 'Directory used to save images.')
    return parser.parse_args()
