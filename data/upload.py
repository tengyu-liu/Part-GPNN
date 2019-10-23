import os

tgt_dir_id = '1-kEapemb4aF47hjj363BJslW2JVqumaT'

src_dir = os.path.join(os.path.dirname(__file__), 'feature_resnet_tengyu')

for fn in os.listdir(src_dir):
    os.system('gdrive-linux-x64 upload --parent %s %s'%(tgt_dir_id, os.path.join(src_dir, fn)))
