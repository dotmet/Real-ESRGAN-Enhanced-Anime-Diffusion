import utils
import os

if __name__ == '__main__':

    utils.check_RealESRGAN()
    args = utils.parse_args()
    
    command = f'\
python repo.py -n {args.n} -w {args.width} -h_ {args.height} \
-g {args.guidance} -gs {args.gen_steps} -s {args.scale} \
-t {args.tile} -e {args.extension} -wd "{args.words}" \
-nwd "{args.neg_words}" -mn "{args.model_name}" \
-o {args.out_dir} -seed {args.seed} -st {args.strength} \
    '
    if args.image != '':
        command += f'-img {args.image}'
        
    status = os.system(command)
    if status==256:
        print('Run failed, try set MKL_THREADING_LAYER=GNU\n')
        os.system(f'export MKL_THREADING_LAYER=GNU && {command}')