import os

camel_exist = [x.strip() for x in open('camel.data').readlines()]

dirname = 'feature_resnet_tengyu'

count = 0
total = len(os.listdir(dirname))

for fn in os.listdir(dirname):
    if fn not in camel_exist:
        os.system('scp %s camel:/mnt/hdd-12t/tengyu/github/Part-GPNN/data/%s'%(os.path.join(dirname, fn), os.path.join(dirname, fn)))
    else:
        print('%s exists in camel'%fn)        
    count += 1
    if count % 10 == 0:
        print('%f%% [%d/%d]'%(count / total * 100, count, total))
